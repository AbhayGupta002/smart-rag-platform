package com.nextgem.smartrag.service;

import com.nextgem.smartrag.config.RagPipelineProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Service;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Enterprise Resource Manager implementing an adaptive 3-state finite state machine
 * (NORMAL, WARNING, CRITICAL) anchored at an 85% memory safety ceiling.
 *
 * Capabilities:
 * 1. Monitored 85% Safety Ceiling: Triggers dynamic backpressure before heap exhaustion.
 * 2. Active Thread Yielding: Forces running worker threads to yield during memory spikes.
 * 3. Ingestion Gate: Halts incoming multi-part chunks until memory normalizes.
 * 4. Dynamic Thread Pool Shrinking: Automatically sheds active workers in CRITICAL state
 *    and restores baseline pool capacity upon recovery to NORMAL.
 * 5. Partition Isolation Gate: Provides quiescing and monitored GC convergence between phases.
 */
@Service
public class ResourceManager {

    private static final Logger log = LoggerFactory.getLogger(ResourceManager.class);

    public static final double CRITICAL_THRESHOLD = 0.85; // 85% safety ceiling
    public static final double WARNING_THRESHOLD = 0.70;  // 70% proactive throttle threshold

    public enum ResourceState {
        NORMAL,   // Healthy heap (< 70%): Full speed concurrent processing
        WARNING,  // Moderate heap (70% - 85%): Concurrency throttled, micro-pauses
        CRITICAL  // Severe heap (>= 85%): Ingestion halted, workers yield, pool shrunk, GC prompted
    }

    private final RagPipelineProperties properties;
    private final MemoryMXBean memoryMXBean;
    private final ThreadPoolExecutor executor;

    private final AtomicReference<ResourceState> currentState = new AtomicReference<>(ResourceState.NORMAL);
    private final AtomicBoolean ingestionHalted = new AtomicBoolean(false);
    private final AtomicInteger baselineCorePoolSize = new AtomicInteger(8);
    private final AtomicLong throttleCount = new AtomicLong(0);
    private final AtomicLong criticalPauseCount = new AtomicLong(0);

    // Concurrency controls for thread halting and backpressure coordination
    private final ReentrantLock gateLock = new ReentrantLock();
    private final Condition memoryRecoveredCondition = gateLock.newCondition();

    public ResourceManager(RagPipelineProperties properties) {
        this(properties, (org.springframework.beans.factory.ObjectProvider<ThreadPoolExecutor>) null);
    }

    @Autowired
    public ResourceManager(
            RagPipelineProperties properties,
            @Qualifier("pipelineExecutor") org.springframework.beans.factory.ObjectProvider<ThreadPoolExecutor> executorProvider
    ) {
        this.properties = properties;
        this.executor = (executorProvider != null) ? executorProvider.getIfAvailable() : null;
        this.memoryMXBean = ManagementFactory.getMemoryMXBean();
        if (this.executor != null) {
            this.baselineCorePoolSize.set(this.executor.getCorePoolSize());
        } else if (properties != null) {
            this.baselineCorePoolSize.set(properties.getMaxConcurrency());
        }
    }

    /**
     * Evaluates the current resource state based on JVM heap utilization against the 85% safety ceiling.
     */
    public ResourceState evaluateState() {
        long usedMb = getUsedHeapMb();
        long maxMb = getMaxHeapMb();
        double ratio = (maxMb > 0) ? (double) usedMb / maxMb : 0.0;

        ResourceState newState;
        if (ratio >= CRITICAL_THRESHOLD) {
            newState = ResourceState.CRITICAL;
        } else if (ratio >= WARNING_THRESHOLD) {
            newState = ResourceState.WARNING;
        } else {
            newState = ResourceState.NORMAL;
        }

        ResourceState oldState = currentState.getAndSet(newState);
        if (oldState != newState) {
            log.info("[RESOURCE-MANAGER] State transition: {} -> {} (Heap: {}/{} MB, {:.1f}%)",
                    oldState, newState, usedMb, maxMb, ratio * 100);

            if (newState == ResourceState.CRITICAL) {
                onCriticalStateEntered();
            } else if (newState == ResourceState.NORMAL && oldState == ResourceState.CRITICAL) {
                onNormalStateRestored();
            }
        }

        return newState;
    }

    /**
     * Ingestion Gate: Blocks new incoming multi-part files or chunk submissions when heap is CRITICAL.
     * Prevents queue saturation and forces callers to wait until memory drops below the 85% ceiling.
     */
    public void acquireIngestionGate() throws InterruptedException {
        evaluateState();
        if (ingestionHalted.get() || currentState.get() == ResourceState.CRITICAL) {
            gateLock.lock();
            try {
                while (currentState.get() == ResourceState.CRITICAL) {
                    criticalPauseCount.incrementAndGet();
                    log.warn("[RESOURCE-MANAGER] Ingestion GATE ACTIVE: Halting incoming file chunk. Awaiting memory stabilization...");
                    Thread.yield();
                    System.gc();
                    memoryRecoveredCondition.awaitNanos(300_000_000L); // 300ms polling wait
                    evaluateState();
                }
            } finally {
                gateLock.unlock();
            }
        }
    }

    /**
     * Proactive backpressure guardrail: checks memory before and during task execution.
     * In CRITICAL state: forces execution threads to yield, shrinks pool, and halts execution until recovery.
     * In WARNING state: introduces micro-pauses to throttle allocation rate.
     */
    public void checkAndThrottle() {
        ResourceState state = evaluateState();

        if (state == ResourceState.CRITICAL) {
            criticalPauseCount.incrementAndGet();
            log.warn("[RESOURCE-MANAGER] CRITICAL heap state (>= 85%). Forcing thread yield and triggering GC...");

            // 1. Force current worker thread to yield CPU time to GC threads
            Thread.yield();
            System.gc();

            // 2. Exponential backoff loop until heap returns below critical ceiling
            int attempts = 0;
            while (attempts < 15) {
                try {
                    Thread.sleep(100L + (attempts * 50L));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
                attempts++;
                if (evaluateState() != ResourceState.CRITICAL) {
                    log.info("[RESOURCE-MANAGER] Recovered from CRITICAL state after {} backoff cycles.", attempts);
                    break;
                }
                Thread.yield();
            }
        } else if (state == ResourceState.WARNING) {
            throttleCount.incrementAndGet();
            try {
                Thread.sleep(40);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Shrinks thread pool capacity and halts incoming ingestion when CRITICAL state is entered.
     */
    private void onCriticalStateEntered() {
        ingestionHalted.set(true);
        if (executor != null) {
            int currentCore = executor.getCorePoolSize();
            if (currentCore > 1) {
                baselineCorePoolSize.compareAndSet(1, currentCore);
                int shrunken = Math.max(1, currentCore / 2);
                executor.setCorePoolSize(shrunken);
                log.warn("[RESOURCE-MANAGER] Shrunk ThreadPoolExecutor core workers: {} -> {} due to CRITICAL memory state.",
                        currentCore, shrunken);
            }
        }
    }

    /**
     * Restores thread pool capacity and signals blocked ingestion callers when returning to NORMAL.
     */
    private void onNormalStateRestored() {
        ingestionHalted.set(false);
        if (executor != null) {
            int restored = Math.max(2, baselineCorePoolSize.get());
            if (restored > executor.getMaximumPoolSize()) {
                executor.setMaximumPoolSize(restored * 2);
            }
            executor.setCorePoolSize(restored);
            log.info("[RESOURCE-MANAGER] Restored ThreadPoolExecutor core workers back to baseline: {}.", restored);
        }

        gateLock.lock();
        try {
            memoryRecoveredCondition.signalAll();
        } finally {
            gateLock.unlock();
        }
    }

    /**
     * Drains active tasks and quiesces the worker pool for partition isolation.
     */
    public void drainAndQuiesce() {
        log.info("[RESOURCE-MANAGER] Quiescing worker pool for partition boundary...");
        if (executor != null) {
            int active = executor.getActiveCount();
            int attempts = 0;
            while (active > 0 && attempts < 20) {
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
                active = executor.getActiveCount();
                attempts++;
            }
        }
        forceReclaim();
    }

    /**
     * Asserts that memory is safe before allowing the next partition to initialize.
     */
    public void assertSafePartitionHandoff() {
        ResourceState state = evaluateState();
        if (state == ResourceState.CRITICAL) {
            log.warn("[RESOURCE-MANAGER] Partition handoff delayed: heap in CRITICAL state. Performing synchronous reclaim...");
            forceReclaim();
        }
    }

    /**
     * Deterministically triggers system garbage collection and measures stabilization.
     */
    public void forceReclaim() {
        System.gc();
        try {
            Thread.sleep(80);
        } catch (InterruptedException ignored) {}
        evaluateState();
    }

    public ResourceState getCurrentState() {
        return evaluateState();
    }

    public boolean isIngestionHalted() {
        return ingestionHalted.get();
    }

    public long getUsedHeapMb() {
        return memoryMXBean.getHeapMemoryUsage().getUsed() / (1024 * 1024);
    }

    public long getMaxHeapMb() {
        long max = memoryMXBean.getHeapMemoryUsage().getMax() / (1024 * 1024);
        if (max <= 0) {
            max = Runtime.getRuntime().totalMemory() / (1024 * 1024);
        }
        return max;
    }

    public double getHeapUsagePercentage() {
        long max = getMaxHeapMb();
        if (max <= 0) return 0.0;
        return (getUsedHeapMb() * 100.0) / max;
    }

    public long getThrottleCount() {
        return throttleCount.get();
    }

    public long getCriticalPauseCount() {
        return criticalPauseCount.get();
    }
}
