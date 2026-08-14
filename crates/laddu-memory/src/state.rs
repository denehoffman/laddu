use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    budget::MemoryBudget,
    discovery::{MemoryProbe, SystemMemoryProbe, discover_host, discover_process_memory},
    error::{MemoryError, MemoryResult},
    pool::{MemoryPool, MemoryPoolInner, ReservationAccount, ResourceLedger},
    report::{MemoryReport, MemoryResourceReport, ProcessMemoryReport},
    resource::{DeviceIdentity, MemoryResource, MemoryResourceKind},
};

/// Live resource discovery and process-wide laddu reservation state.
#[derive(Clone, Debug)]
pub struct MemoryState {
    inner: Arc<MemoryStateInner>,
}

#[derive(Debug)]
pub(crate) struct MemoryStateInner {
    pub(crate) resources: Mutex<BTreeMap<String, ResourceLedger>>,
    process_high_water: AtomicU64,
}

impl MemoryState {
    /// Discovers host memory and creates an independent reservation state.
    pub fn discover() -> Self {
        let host = discover_host();
        let mut resources = BTreeMap::new();
        resources.insert(host.id.clone(), ResourceLedger::new(host));
        Self {
            inner: Arc::new(MemoryStateInner {
                resources: Mutex::new(resources),
                process_high_water: AtomicU64::new(0),
            }),
        }
    }

    /// Returns the process-wide default state.
    pub fn current() -> Self {
        static CURRENT: OnceLock<MemoryState> = OnceLock::new();
        CURRENT.get_or_init(Self::discover).clone()
    }

    /// Refreshes host total and available memory.
    pub fn refresh(&self) {
        self.refresh_inner();
    }
    fn refresh_inner(&self) -> Option<ProcessMemoryReport> {
        self.refresh_with_probe(&SystemMemoryProbe)
    }

    fn refresh_with_probe(&self, probe: &dyn MemoryProbe) -> Option<ProcessMemoryReport> {
        let host = discover_host();
        let process = self.sample_process_memory();
        let targets = {
            let mut resources = self
                .inner
                .resources
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let ledger = resources
                .entry(host.id.clone())
                .or_insert_with(|| ResourceLedger::new(host.clone()));
            ledger.update_snapshot(host);
            resources
                .values()
                .filter(|ledger| ledger.snapshot.is_refreshable())
                .map(|ledger| ledger.snapshot.clone())
                .collect::<Vec<_>>()
        };
        let outcomes = targets
            .into_iter()
            .map(|target| {
                let outcome = probe.probe_device(&target);
                (target, outcome)
            })
            .collect::<Vec<_>>();
        let mut resources = self
            .inner
            .resources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (target, outcome) in outcomes {
            if let Ok(snapshot) = outcome
                && let Some(ledger) = resources.get_mut(&target.id)
            {
                ledger.apply_telemetry(&target, snapshot);
            }
        }
        process
    }

    /// Returns the current host snapshot.
    pub fn host(&self) -> MemoryResource {
        self.resource("host").unwrap_or_else(discover_host)
    }

    /// Registers capacity telemetry for a runtime-selected accelerator.
    ///
    /// The runtime owns the stable identifier and adapter identity. Platform
    /// telemetry is used when available; otherwise `fallback_bytes` becomes an
    /// adaptive capacity estimate.
    pub fn register_discovered_device(
        &self,
        id: impl Into<String>,
        name: impl Into<String>,
        identity: DeviceIdentity,
        fallback_bytes: u64,
    ) {
        self.insert_device_snapshot(MemoryResource::discover_device(
            id,
            name,
            identity,
            fallback_bytes,
        ));
    }

    /// Overrides capacity telemetry for a device until another user override.
    pub fn override_device_capacity(
        &self,
        id: impl Into<String>,
        name: impl Into<String>,
        total_bytes: u64,
        available_bytes: Option<u64>,
    ) {
        self.insert_device_snapshot(MemoryResource::user_device(
            id,
            name,
            total_bytes,
            available_bytes,
        ));
    }

    pub(crate) fn insert_device_snapshot(&self, resource: MemoryResource) {
        let mut resources = self
            .inner
            .resources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(ledger) = resources.get_mut(&resource.id) {
            ledger.update_snapshot(resource);
        } else {
            resources.insert(resource.id.clone(), ResourceLedger::new(resource));
        }
    }

    /// Returns one resource by stable identifier.
    pub fn resource(&self, id: &str) -> Option<MemoryResource> {
        self.inner
            .resources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .map(|ledger| ledger.snapshot.clone())
    }

    /// Returns all registered accelerator resources.
    pub fn devices(&self) -> Vec<MemoryResource> {
        self.inner
            .resources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|ledger| ledger.snapshot.kind == MemoryResourceKind::Device)
            .map(|ledger| ledger.snapshot.clone())
            .collect()
    }

    /// Resolves a budget and creates a local pool for a resource.
    ///
    /// # Errors
    ///
    /// Returns an error when the resource is unknown or the budget cannot be resolved.
    pub fn pool(&self, resource_id: &str, budget: MemoryBudget) -> MemoryResult<MemoryPool> {
        let resource = self.resource(resource_id).ok_or_else(|| {
            MemoryError::InvalidBudget(format!("unknown memory resource {resource_id:?}"))
        })?;
        resource.validate().map_err(|error| {
            MemoryError::InvalidBudget(format!(
                "invalid memory resource {resource_id:?}: {error:?}"
            ))
        })?;
        let capacity = budget.resolve(&resource)?;
        Ok(MemoryPool {
            inner: Arc::new(MemoryPoolInner {
                requested: budget,
                accounting: Arc::new(ReservationAccount::new(
                    Arc::downgrade(&self.inner),
                    resource_id.to_owned(),
                    capacity,
                )),
            }),
        })
    }

    /// Returns a structured report for all resources.
    pub fn report(&self) -> MemoryReport {
        let process = self.refresh_inner();
        let resources = self
            .inner
            .resources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        MemoryReport {
            process,
            resources: resources
                .values()
                .map(|ledger| MemoryResourceReport {
                    resource: ledger.snapshot.clone(),
                    laddu_reserved_bytes: ledger.reserved,
                    laddu_high_water_bytes: ledger.high_water,
                })
                .collect(),
        }
    }

    fn sample_process_memory(&self) -> Option<ProcessMemoryReport> {
        let snapshot = discover_process_memory().ok()?;
        self.inner
            .process_high_water
            .fetch_max(snapshot.resident_bytes, Ordering::AcqRel);
        Some(ProcessMemoryReport {
            resident_bytes: snapshot.resident_bytes,
            virtual_bytes: snapshot.virtual_bytes,
            sampled_high_water_bytes: self.inner.process_high_water.load(Ordering::Acquire),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapacitySource, DeviceIdentity, MemoryResourceKind,
        discovery::{CapacitySnapshot, ProbeFailure, ProbeOutcome},
    };
    use std::{sync::mpsc, thread, time::Duration};

    fn resource() -> MemoryResource {
        MemoryResource {
            id: "test".into(),
            name: "Test".into(),
            kind: MemoryResourceKind::Device,
            total_bytes: Some(1_000),
            available_bytes: Some(500),
            capacity_source: CapacitySource::User,
            device_identity: None,
        }
    }
    fn refreshable(identity: usize) -> MemoryResource {
        MemoryResource {
            capacity_source: CapacitySource::Adaptive,
            device_identity: Some(DeviceIdentity {
                adapter_index: identity,
                vendor_id: 1,
                device_id: 2,
                pci_bus_id: format!("bus-{identity}"),
            }),
            ..resource()
        }
    }
    fn snapshot(total_bytes: u64, available_bytes: u64) -> CapacitySnapshot {
        CapacitySnapshot {
            total_bytes,
            available_bytes,
            source: CapacitySource::Drm,
        }
    }

    #[derive(Clone, Copy)]
    struct FixedProbe(ProbeOutcome);
    impl MemoryProbe for FixedProbe {
        fn probe_device(&self, _: &MemoryResource) -> ProbeOutcome {
            self.0
        }
    }
    struct BlockingProbe {
        entered: mpsc::SyncSender<()>,
        release: Mutex<mpsc::Receiver<()>>,
        outcome: ProbeOutcome,
    }
    impl MemoryProbe for BlockingProbe {
        fn probe_device(&self, _: &MemoryResource) -> ProbeOutcome {
            self.entered.send(()).unwrap();
            self.release.lock().unwrap().recv().unwrap();
            self.outcome
        }
    }
    struct ReplacingProbe {
        state: MemoryState,
        replacement: MemoryResource,
        outcome: ProbeOutcome,
    }
    impl MemoryProbe for ReplacingProbe {
        fn probe_device(&self, _: &MemoryResource) -> ProbeOutcome {
            self.state.insert_device_snapshot(self.replacement.clone());
            self.outcome
        }
    }

    #[test]
    fn ledgers_clamp_capacity_and_keep_user_overrides_sticky() {
        let state = MemoryState::discover();
        let discovered = MemoryResource {
            total_bytes: Some(1_000),
            available_bytes: Some(1_100),
            ..refreshable(0)
        };
        state.insert_device_snapshot(discovered);
        assert_eq!(state.resource("test").unwrap().available_bytes, Some(1_000));
        let user = MemoryResource::user_device("test", "Test", 700, Some(650));
        state.override_device_capacity("test", "Test", 700, Some(650));
        state.insert_device_snapshot(MemoryResource {
            total_bytes: Some(900),
            available_bytes: Some(800),
            ..refreshable(0)
        });
        assert_eq!(state.resource("test"), Some(user));

        let replacement = MemoryResource::user_device("test", "Test", 600, Some(550));
        state.override_device_capacity("test", "Test", 600, Some(550));
        assert_eq!(state.resource("test"), Some(replacement));
        assert_eq!(
            state
                .devices()
                .into_iter()
                .filter(|resource| resource.id == "test")
                .count(),
            1
        );
    }

    #[test]
    fn refresh_probes_without_holding_the_resource_lock() {
        let state = MemoryState::discover();
        state.insert_device_snapshot(refreshable(0));
        let (entered_tx, entered_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::sync_channel(0);
        let probe = BlockingProbe {
            entered: entered_tx,
            release: Mutex::new(release_rx),
            outcome: Ok(snapshot(900, 700)),
        };
        let refresh_state = state.clone();
        let refresh = thread::spawn(move || refresh_state.refresh_with_probe(&probe));
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let access_state = state.clone();
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let access = thread::spawn(move || {
            assert!(access_state.resource("test").is_some());
            done_tx.send(()).unwrap();
        });
        let result = done_rx.recv_timeout(Duration::from_secs(2));
        release_tx.send(()).unwrap();
        refresh.join().unwrap();
        access.join().unwrap();
        assert!(result.is_ok(), "resource access blocked during telemetry");
    }

    #[test]
    fn refresh_retains_stale_snapshot_on_failure() {
        let state = MemoryState::discover();
        state.insert_device_snapshot(refreshable(0));
        state.refresh_with_probe(&FixedProbe(Ok(snapshot(800, 600))));
        let refreshed = state.resource("test").unwrap();
        state.refresh_with_probe(&FixedProbe(Err(ProbeFailure::Unavailable)));
        assert_eq!(state.resource("test"), Some(refreshed));
    }

    #[test]
    fn refresh_does_not_overwrite_user_override_or_replaced_device() {
        let state = MemoryState::discover();
        state.insert_device_snapshot(refreshable(0));
        let user = MemoryResource::user_device("test", "Test", 700, Some(650));
        state.refresh_with_probe(&ReplacingProbe {
            state: state.clone(),
            replacement: user.clone(),
            outcome: Ok(snapshot(900, 800)),
        });
        assert_eq!(state.resource("test"), Some(user));

        let state = MemoryState::discover();
        state.insert_device_snapshot(refreshable(0));
        let replacement = MemoryResource {
            total_bytes: Some(400),
            available_bytes: Some(300),
            ..refreshable(1)
        };
        state.refresh_with_probe(&ReplacingProbe {
            state: state.clone(),
            replacement: replacement.clone(),
            outcome: Ok(snapshot(900, 800)),
        });
        assert_eq!(state.resource("test"), Some(replacement));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn reports_current_process_memory() {
        let state = MemoryState::discover();
        let first = state.report().process.unwrap();
        let second = state.report().process.unwrap();
        assert!(first.resident_bytes > 0);
        assert!(first.virtual_bytes >= first.resident_bytes);
        assert!(second.sampled_high_water_bytes >= second.resident_bytes);
    }
}
