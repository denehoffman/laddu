//! Memory discovery, budgeting, reservation, and reporting for laddu.

use std::{
    collections::BTreeMap,
    fmt,
    str::FromStr,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, get_current_pid};
use thiserror::Error;

const AUTO_AVAILABLE_FRACTION: f64 = 0.80;

/// Result type for memory planning operations.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Errors produced while discovering or reserving memory.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MemoryError {
    /// A budget string or percentage is invalid.
    #[error("invalid memory budget: {0}")]
    InvalidBudget(String),
    /// A percentage cannot be resolved because capacity telemetry is unavailable.
    #[error("cannot resolve {budget} for {resource}: {basis} memory is unavailable")]
    UnknownCapacity {
        /// Resource label.
        resource: String,
        /// Requested budget.
        budget: MemoryBudget,
        /// Missing capacity basis.
        basis: &'static str,
    },
    /// A reservation exceeds the effective pool limit.
    #[error(
        "memory budget exceeded for {resource}: requested {requested} bytes, \
         {remaining} bytes remain"
    )]
    BudgetExceeded {
        /// Resource label.
        resource: String,
        /// Requested reservation.
        requested: u64,
        /// Remaining reservable bytes.
        remaining: u64,
    },
}

/// The basis used to obtain a resource's capacity information.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacitySource {
    /// Host operating-system telemetry.
    OperatingSystem,
    /// A process or container limit.
    Cgroup,
    /// NVIDIA Management Library telemetry.
    Nvml,
    /// Linux DRM/sysfs telemetry.
    Drm,
    /// Windows DXGI telemetry.
    Dxgi,
    /// Apple Metal working-set telemetry.
    Metal,
    /// Capacity supplied by the user.
    User,
    /// Capacity is not observable and planning is adaptive.
    Adaptive,
}

/// Kind of physical memory resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryResourceKind {
    /// Host RAM.
    Host,
    /// Accelerator-local or unified memory.
    Device,
}

/// A requested memory limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBudget {
    /// Automatically use 80% of currently available memory.
    #[default]
    Auto,
    /// An absolute number of bytes.
    Bytes(u64),
    /// A fraction in `(0, 1]` of total physical capacity.
    PercentTotal(f64),
    /// A fraction in `(0, 1]` of currently available capacity.
    PercentAvailable(f64),
}

impl MemoryBudget {
    /// Creates an absolute byte budget.
    pub const fn bytes(bytes: u64) -> Self {
        Self::Bytes(bytes)
    }

    /// Creates a percentage-of-total budget.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::InvalidBudget`] unless `percent` is in `(0, 100]`.
    pub fn percent_total(percent: f64) -> MemoryResult<Self> {
        Ok(Self::PercentTotal(validate_percent(percent)? / 100.0))
    }

    /// Creates a percentage-of-available budget.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::InvalidBudget`] unless `percent` is in `(0, 100]`.
    pub fn percent_available(percent: f64) -> MemoryResult<Self> {
        Ok(Self::PercentAvailable(validate_percent(percent)? / 100.0))
    }

    /// Resolves this request for a resource snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error for zero budgets, invalid percentages, or unavailable
    /// capacity telemetry.
    pub fn resolve(self, resource: &MemoryResource) -> MemoryResult<u64> {
        let resolved = match self {
            Self::Auto => resource
                .available_bytes
                .map(|bytes| scaled_bytes(bytes, AUTO_AVAILABLE_FRACTION))
                .or(resource.total_bytes.map(|bytes| scaled_bytes(bytes, 0.5)))
                .ok_or_else(|| MemoryError::UnknownCapacity {
                    resource: resource.name.clone(),
                    budget: self,
                    basis: "available",
                })?,
            Self::Bytes(bytes) => bytes,
            Self::PercentTotal(fraction) => {
                validate_fraction(fraction)?;
                scaled_bytes(
                    resource
                        .total_bytes
                        .ok_or_else(|| MemoryError::UnknownCapacity {
                            resource: resource.name.clone(),
                            budget: self,
                            basis: "total",
                        })?,
                    fraction,
                )
            }
            Self::PercentAvailable(fraction) => {
                validate_fraction(fraction)?;
                scaled_bytes(
                    resource
                        .available_bytes
                        .ok_or_else(|| MemoryError::UnknownCapacity {
                            resource: resource.name.clone(),
                            budget: self,
                            basis: "available",
                        })?,
                    fraction,
                )
            }
        };
        if resolved == 0 {
            return Err(MemoryError::InvalidBudget(
                "resolved budget must be greater than zero".into(),
            ));
        }
        Ok(resolved)
    }
}

impl fmt::Display for MemoryBudget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::Bytes(bytes) => write!(formatter, "{bytes} B"),
            Self::PercentTotal(value) => write!(formatter, "{}% total", value * 100.0),
            Self::PercentAvailable(value) => {
                write!(formatter, "{}% available", value * 100.0)
            }
        }
    }
}

impl FromStr for MemoryBudget {
    type Err = MemoryError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let normalized = input.trim().to_ascii_lowercase();
        if normalized == "auto" {
            return Ok(Self::Auto);
        }
        if let Some((percent, suffix)) = normalized.split_once('%') {
            let percent = percent
                .trim()
                .parse::<f64>()
                .map_err(|_| MemoryError::InvalidBudget(input.into()))?;
            let suffix = suffix.trim();
            return match suffix {
                "" | "total" => Self::percent_total(percent),
                "available" | "free" | "remaining" => Self::percent_available(percent),
                _ => Err(MemoryError::InvalidBudget(input.into())),
            };
        }
        parse_bytes(&normalized).map(Self::Bytes)
    }
}

/// Host and optional accelerator budgets for one execution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    /// Host allocations, including source and staging buffers.
    pub host: MemoryBudget,
    /// Device allocations. Required only for accelerator execution.
    pub device: Option<MemoryBudget>,
}

impl Default for MemoryPlan {
    fn default() -> Self {
        Self {
            host: MemoryBudget::Auto,
            device: Some(MemoryBudget::Auto),
        }
    }
}

impl MemoryPlan {
    /// Creates a host-only plan.
    pub const fn host(host: MemoryBudget) -> Self {
        Self { host, device: None }
    }

    /// Creates a host-and-device plan.
    pub const fn host_device(host: MemoryBudget, device: MemoryBudget) -> Self {
        Self {
            host,
            device: Some(device),
        }
    }
}

/// Stable information used to match a runtime accelerator to platform telemetry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceIdentity {
    /// Backend adapter index.
    pub adapter_index: usize,
    /// PCI vendor identifier, or zero when unavailable.
    pub vendor_id: u32,
    /// PCI device identifier, or zero when unavailable.
    pub device_id: u32,
    /// PCI bus identifier, or an empty string when unavailable.
    pub pci_bus_id: String,
}

/// Snapshot of one physical memory resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryResource {
    /// Stable resource identifier.
    pub id: String,
    /// Human-readable resource name.
    pub name: String,
    /// Host or accelerator memory.
    pub kind: MemoryResourceKind,
    /// Total capacity when observable.
    pub total_bytes: Option<u64>,
    /// Currently available capacity when observable.
    pub available_bytes: Option<u64>,
    /// Source of capacity information.
    pub capacity_source: CapacitySource,
    /// Accelerator identity used to refresh platform telemetry.
    pub device_identity: Option<DeviceIdentity>,
}

impl MemoryResource {
    /// Creates an accelerator resource whose physical capacity is unavailable.
    pub fn adaptive_device(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind: MemoryResourceKind::Device,
            total_bytes: None,
            available_bytes: None,
            capacity_source: CapacitySource::Adaptive,
            device_identity: None,
        }
    }

    /// Discovers accelerator capacity from platform telemetry, falling back to
    /// an adaptive backend limit when dedicated-memory telemetry is absent.
    pub fn discover_device(
        id: impl Into<String>,
        name: impl Into<String>,
        identity: DeviceIdentity,
        fallback_bytes: u64,
    ) -> Self {
        let mut resource = Self::adaptive_device(id, name);
        resource.device_identity = Some(identity);
        if let Some((total, available, source)) = refresh_device_memory(&resource) {
            resource.total_bytes = Some(total);
            resource.available_bytes = Some(available);
            resource.capacity_source = source;
            return resource;
        }
        resource.total_bytes = Some(fallback_bytes);
        resource.available_bytes = Some(fallback_bytes);
        resource
    }

    /// Creates a resource with an explicit user-provided capacity.
    pub fn with_capacity(mut self, total_bytes: u64, available_bytes: Option<u64>) -> Self {
        self.total_bytes = Some(total_bytes);
        self.available_bytes = Some(available_bytes.unwrap_or(total_bytes).min(total_bytes));
        self.capacity_source = CapacitySource::User;
        self
    }

    /// Creates a budget bound to this resource.
    pub const fn budget(&self, budget: MemoryBudget) -> MemoryBudget {
        budget
    }
}

#[derive(Debug)]
struct ResourceLedger {
    snapshot: MemoryResource,
    reserved: u64,
    high_water: u64,
}

impl ResourceLedger {
    fn try_reserve(&mut self, bytes: u64) -> MemoryResult<()> {
        let physical_limit = self
            .snapshot
            .available_bytes
            .or(self.snapshot.total_bytes)
            .unwrap_or(u64::MAX);
        let Some(next) = self.reserved.checked_add(bytes) else {
            return Err(self.budget_exceeded(bytes, physical_limit));
        };
        if next > physical_limit {
            return Err(self.budget_exceeded(bytes, physical_limit));
        }
        self.reserved = next;
        self.high_water = self.high_water.max(next);
        Ok(())
    }

    fn release(&mut self, bytes: u64) {
        self.reserved = self.reserved.saturating_sub(bytes);
    }

    fn budget_exceeded(&self, requested: u64, physical_limit: u64) -> MemoryError {
        MemoryError::BudgetExceeded {
            resource: self.snapshot.name.clone(),
            requested,
            remaining: physical_limit.saturating_sub(self.reserved),
        }
    }
}

/// Live resource discovery and process-wide laddu reservation state.
#[derive(Clone, Debug)]
pub struct MemoryState {
    inner: Arc<MemoryStateInner>,
}

#[derive(Debug)]
struct MemoryStateInner {
    resources: Mutex<BTreeMap<String, ResourceLedger>>,
    process_high_water: AtomicU64,
}

impl MemoryState {
    /// Discovers host memory and creates an independent reservation state.
    pub fn discover() -> Self {
        let host = discover_host();
        let mut resources = BTreeMap::new();
        resources.insert(
            host.id.clone(),
            ResourceLedger {
                snapshot: host,
                reserved: 0,
                high_water: 0,
            },
        );
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
        let host = discover_host();
        let process = self.sample_process_memory();
        let mut resources = self
            .inner
            .resources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ledger = resources
            .entry(host.id.clone())
            .or_insert_with(|| ResourceLedger {
                snapshot: host.clone(),
                reserved: 0,
                high_water: 0,
            });
        ledger.snapshot = host;
        for ledger in resources.values_mut() {
            if ledger.snapshot.kind != MemoryResourceKind::Device
                || ledger.snapshot.capacity_source == CapacitySource::User
            {
                continue;
            }
            if let Some((total, available, source)) = refresh_device_memory(&ledger.snapshot) {
                ledger.snapshot.total_bytes = Some(total);
                ledger.snapshot.available_bytes = Some(available);
                ledger.snapshot.capacity_source = source;
            }
        }
        process
    }

    /// Returns the current host snapshot.
    pub fn host(&self) -> MemoryResource {
        self.resource("host").unwrap_or_else(discover_host)
    }

    /// Registers or refreshes an accelerator resource.
    pub fn register_device(&self, resource: MemoryResource) {
        let mut resources = self
            .inner
            .resources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ledger = resources
            .entry(resource.id.clone())
            .or_insert_with(|| ResourceLedger {
                snapshot: resource.clone(),
                reserved: 0,
                high_water: 0,
            });
        // A user capacity override is authoritative until explicitly replaced
        // by another user override.
        if ledger.snapshot.capacity_source != CapacitySource::User
            || resource.capacity_source == CapacitySource::User
        {
            ledger.snapshot = resource;
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
        let capacity = budget.resolve(&resource)?;
        Ok(MemoryPool {
            inner: Arc::new(MemoryPoolInner {
                requested: budget,
                accounting: Arc::new(ReservationAccount {
                    state: Arc::downgrade(&self.inner),
                    resource_id: resource_id.to_owned(),
                    capacity,
                    reserved: AtomicU64::new(0),
                    high_water: AtomicU64::new(0),
                }),
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
        let (resident_bytes, virtual_bytes) = discover_process_memory()?;
        self.inner
            .process_high_water
            .fetch_max(resident_bytes, Ordering::AcqRel);
        Some(ProcessMemoryReport {
            resident_bytes,
            virtual_bytes,
            sampled_high_water_bytes: self.inner.process_high_water.load(Ordering::Acquire),
        })
    }
}

/// A resolved, reservable limit within one physical resource.
#[derive(Clone, Debug)]
pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

#[derive(Debug)]
struct MemoryPoolInner {
    requested: MemoryBudget,
    accounting: Arc<ReservationAccount>,
}

#[derive(Debug)]
struct ReservationAccount {
    state: Weak<MemoryStateInner>,
    resource_id: String,
    capacity: u64,
    reserved: AtomicU64,
    high_water: AtomicU64,
}

impl ReservationAccount {
    fn try_reserve(self: &Arc<Self>, bytes: u64) -> MemoryResult<ReservationToken> {
        let state = self.state.upgrade();
        let mut resources = state
            .as_ref()
            .map(|state| state.resources.lock().unwrap_or_else(|e| e.into_inner()));

        let next = self.try_reserve_local(bytes)?;
        if let Some(ledger) = resources
            .as_mut()
            .and_then(|resources| resources.get_mut(&self.resource_id))
            && let Err(error) = ledger.try_reserve(bytes)
        {
            self.release_local(bytes);
            return Err(error);
        }

        update_max(&self.high_water, next);
        Ok(ReservationToken {
            account: Arc::clone(self),
            bytes,
        })
    }

    fn try_reserve_local(&self, bytes: u64) -> MemoryResult<u64> {
        self.reserved
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|&next| next <= self.capacity)
            })
            .map(|previous| previous + bytes)
            .map_err(|_| self.budget_exceeded(bytes))
    }

    fn release(&self, bytes: u64) {
        if let Some(state) = self.state.upgrade() {
            let mut resources = state.resources.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(ledger) = resources.get_mut(&self.resource_id) {
                ledger.release(bytes);
            }
        }
        self.release_local(bytes);
    }

    fn release_local(&self, bytes: u64) {
        self.reserved.fetch_sub(bytes, Ordering::AcqRel);
    }

    fn budget_exceeded(&self, requested: u64) -> MemoryError {
        MemoryError::BudgetExceeded {
            resource: self.resource_id.clone(),
            requested,
            remaining: self
                .capacity
                .saturating_sub(self.reserved.load(Ordering::Acquire)),
        }
    }
}

#[derive(Debug)]
struct ReservationToken {
    account: Arc<ReservationAccount>,
    bytes: u64,
}

impl Drop for ReservationToken {
    fn drop(&mut self) {
        self.account.release(self.bytes);
    }
}

impl MemoryPool {
    /// Requested budget specification.
    pub fn requested(&self) -> MemoryBudget {
        self.inner.requested
    }

    /// Resolved pool capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.inner.accounting.capacity
    }

    /// Currently reserved bytes.
    pub fn reserved(&self) -> u64 {
        self.inner.accounting.reserved.load(Ordering::Acquire)
    }

    /// Remaining reservable bytes.
    pub fn remaining(&self) -> u64 {
        self.capacity().saturating_sub(self.reserved())
    }

    /// Highest concurrent reservation observed by this pool.
    pub fn high_water(&self) -> u64 {
        self.inner.accounting.high_water.load(Ordering::Acquire)
    }

    /// Attempts to reserve `bytes` until the returned lease is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::BudgetExceeded`] if the local or shared resource
    /// limit lacks sufficient remaining capacity.
    pub fn reserve(&self, bytes: u64) -> MemoryResult<MemoryLease> {
        Ok(MemoryLease {
            inner: Arc::new(MemoryLeaseInner {
                reservation: self.inner.accounting.try_reserve(bytes)?,
            }),
        })
    }

    /// Returns a snapshot of this pool's planning and usage.
    pub fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            resource_id: self.inner.accounting.resource_id.clone(),
            requested: self.requested(),
            effective_bytes: self.capacity(),
            reserved_bytes: self.reserved(),
            remaining_bytes: self.remaining(),
            high_water_bytes: self.high_water(),
        }
    }
}

/// A live memory reservation. Clones share one reservation.
#[derive(Clone, Debug)]
pub struct MemoryLease {
    inner: Arc<MemoryLeaseInner>,
}

#[derive(Debug)]
struct MemoryLeaseInner {
    reservation: ReservationToken,
}

impl MemoryLease {
    /// Reserved byte count.
    pub fn bytes(&self) -> u64 {
        self.inner.reservation.bytes
    }
}

/// Report for one resolved pool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryPoolReport {
    /// Stable resource identifier.
    pub resource_id: String,
    /// Requested budget.
    pub requested: MemoryBudget,
    /// Resolved capacity.
    pub effective_bytes: u64,
    /// Currently reserved bytes.
    pub reserved_bytes: u64,
    /// Remaining bytes.
    pub remaining_bytes: u64,
    /// Highest concurrent reservation.
    pub high_water_bytes: u64,
}

/// Report for one physical resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryResourceReport {
    /// Resource snapshot.
    pub resource: MemoryResource,
    /// Currently reserved by laddu.
    pub laddu_reserved_bytes: u64,
    /// Process-state high-water reservation.
    pub laddu_high_water_bytes: u64,
}

/// Report covering all resources in a [`MemoryState`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReport {
    /// Current process memory sampled when the report was generated.
    pub process: Option<ProcessMemoryReport>,
    /// Physical resource reports.
    pub resources: Vec<MemoryResourceReport>,
}

/// Sampled operating-system memory counters for the current process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMemoryReport {
    /// Current resident-set size.
    pub resident_bytes: u64,
    /// Current virtual-memory size.
    pub virtual_bytes: u64,
    /// Largest resident-set size observed by this [`MemoryState`].
    ///
    /// This is a sampled high-water mark, not an operating-system lifetime peak.
    pub sampled_high_water_bytes: u64,
}

/// One memory-derived execution decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryDecision {
    /// Operation or dataset label.
    pub label: String,
    /// Fixed bytes required regardless of event count.
    pub fixed_bytes: u64,
    /// Estimated incremental bytes per event.
    pub bytes_per_event: u64,
    /// Chosen internal event count.
    pub chunk_events: usize,
    /// Estimated peak tracked bytes.
    pub estimated_peak_bytes: u64,
    /// Actual tracked high-water bytes when known.
    pub actual_high_water_bytes: Option<u64>,
    /// Selected storage/execution strategy.
    pub strategy: String,
}

impl MemoryDecision {
    /// Derives the largest nonzero event chunk fitting within `available_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::BudgetExceeded`] when fixed cost plus one event
    /// cannot fit.
    pub fn fit(
        label: impl Into<String>,
        fixed_bytes: u64,
        bytes_per_event: u64,
        available_bytes: u64,
        event_limit: usize,
        strategy: impl Into<String>,
    ) -> MemoryResult<Self> {
        let label = label.into();
        let per_event = bytes_per_event.max(1);
        let capacity = available_bytes.saturating_sub(fixed_bytes);
        let events = usize::try_from(capacity / per_event)
            .unwrap_or(usize::MAX)
            .min(event_limit);
        if event_limit > 0 && events == 0 {
            return Err(MemoryError::BudgetExceeded {
                resource: label,
                requested: fixed_bytes.saturating_add(per_event),
                remaining: available_bytes,
            });
        }
        let peak = fixed_bytes.saturating_add(per_event.saturating_mul(events as u64));
        Ok(Self {
            label,
            fixed_bytes,
            bytes_per_event,
            chunk_events: events,
            estimated_peak_bytes: peak,
            actual_high_water_bytes: None,
            strategy: strategy.into(),
        })
    }
}

fn discover_host() -> MemoryResource {
    let mut system = System::new();
    system.refresh_memory();
    let mut total = system.total_memory();
    let mut available = system.available_memory();
    let mut capacity_source = CapacitySource::OperatingSystem;
    if let Some((limit, available_in_group)) = discover_cgroup_memory()
        && limit < total
    {
        total = limit;
        available = available.min(available_in_group);
        capacity_source = CapacitySource::Cgroup;
    }
    MemoryResource {
        id: "host".into(),
        name: "Host memory".into(),
        kind: MemoryResourceKind::Host,
        total_bytes: Some(total),
        available_bytes: Some(available),
        capacity_source,
        device_identity: None,
    }
}

fn discover_cgroup_memory() -> Option<(u64, u64)> {
    let pid = get_current_pid().ok()?;
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let limits = system.process(pid)?.cgroup_limits()?;
    Some((limits.total_memory, limits.free_memory))
}

fn discover_process_memory() -> Option<(u64, u64)> {
    let pid = get_current_pid().ok()?;
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let process = system.process(pid)?;
    Some((process.memory(), process.virtual_memory()))
}

fn refresh_device_memory(resource: &MemoryResource) -> Option<(u64, u64, CapacitySource)> {
    let identity = resource.device_identity.as_ref()?;
    #[cfg(feature = "nvml")]
    if let Some((total, available)) = discover_nvml_memory(&identity.pci_bus_id) {
        return Some((total, available, CapacitySource::Nvml));
    }
    #[cfg(target_os = "windows")]
    if let Some((total, available)) = discover_dxgi_memory(identity) {
        return Some((total, available, CapacitySource::Dxgi));
    }
    #[cfg(target_os = "macos")]
    if let Some((total, available)) = discover_metal_memory(identity, &resource.name) {
        return Some((total, available, CapacitySource::Metal));
    }
    #[cfg(target_os = "linux")]
    if let Some((total, available)) = discover_drm_memory(&identity.pci_bus_id) {
        return Some((total, available, CapacitySource::Drm));
    }
    None
}

#[cfg(feature = "nvml")]
fn discover_nvml_memory(pci_bus_id: &str) -> Option<(u64, u64)> {
    if pci_bus_id.is_empty() {
        return None;
    }
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let device = nvml.device_by_pci_bus_id(pci_bus_id).ok()?;
    let memory = device.memory_info().ok()?;
    Some((memory.total, memory.free))
}

#[cfg(target_os = "windows")]
fn discover_dxgi_memory(identity: &DeviceIdentity) -> Option<(u64, u64)> {
    use windows::{
        Win32::Graphics::Dxgi::{
            CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
            IDXGIAdapter3, IDXGIFactory1,
        },
        core::Interface,
    };

    // SAFETY: DXGI factory and adapter methods own their returned COM interfaces,
    // and all output pointers are provided by the windows crate.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut fallback = None;
        for index in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(index) else {
                break;
            };
            let Ok(description) = adapter.GetDesc1() else {
                continue;
            };
            if description.VendorId != identity.vendor_id
                || description.DeviceId != identity.device_id
            {
                continue;
            }
            let adapter: IDXGIAdapter3 = adapter.cast().ok()?;
            let mut memory = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut memory)
                .ok()?;
            let total = memory.Budget;
            if total == 0 {
                continue;
            }
            let snapshot = (total, total.saturating_sub(memory.CurrentUsage));
            if index as usize == identity.adapter_index {
                return Some(snapshot);
            }
            fallback.get_or_insert(snapshot);
        }
        return fallback;
    }
}

#[cfg(target_os = "macos")]
fn discover_metal_memory(identity: &DeviceIdentity, expected_name: &str) -> Option<(u64, u64)> {
    use objc2_metal::MTLDevice;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    let devices = objc2_metal::MTLCopyAllDevices();
    let device = (0..devices.count())
        .map(|index| devices.objectAtIndex(index))
        .find(|device| device.name().to_string() == expected_name)
        .or_else(|| {
            (identity.adapter_index < devices.count())
                .then(|| devices.objectAtIndex(identity.adapter_index))
        })?;
    let total = device.recommendedMaxWorkingSetSize();
    let used = device.currentAllocatedSize() as u64;
    (total > 0).then_some((total, total.saturating_sub(used)))
}

#[cfg(target_os = "linux")]
fn discover_drm_memory(pci_bus_id: &str) -> Option<(u64, u64)> {
    if pci_bus_id.is_empty() {
        return None;
    }
    let entries = std::fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("card") || name.to_string_lossy().contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let Ok(uevent) = std::fs::read_to_string(device.join("uevent")) else {
            continue;
        };
        let matches_device = uevent.lines().any(|line| {
            line.strip_prefix("PCI_SLOT_NAME=")
                .is_some_and(|slot| slot.eq_ignore_ascii_case(pci_bus_id))
        });
        if !matches_device {
            continue;
        }
        let Some(total) = read_sysfs_u64(device.join("mem_info_vram_total")) else {
            continue;
        };
        let used = read_sysfs_u64(device.join("mem_info_vram_used")).unwrap_or(0);
        return Some((total, total.saturating_sub(used)));
    }
    None
}

#[cfg(target_os = "linux")]
fn read_sysfs_u64(path: impl AsRef<std::path::Path>) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn validate_percent(percent: f64) -> MemoryResult<f64> {
    if percent.is_finite() && percent > 0.0 && percent <= 100.0 {
        Ok(percent)
    } else {
        Err(MemoryError::InvalidBudget(
            "percentage must be finite and in (0, 100]".into(),
        ))
    }
}

fn validate_fraction(fraction: f64) -> MemoryResult<()> {
    validate_percent(fraction * 100.0).map(|_| ())
}

fn scaled_bytes(bytes: u64, fraction: f64) -> u64 {
    ((bytes as f64) * fraction).floor().min(u64::MAX as f64) as u64
}

fn parse_bytes(input: &str) -> MemoryResult<u64> {
    let split = input
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(input.len());
    let (number, unit) = input.split_at(split);
    let value = number
        .trim()
        .parse::<f64>()
        .map_err(|_| MemoryError::InvalidBudget(input.into()))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(MemoryError::InvalidBudget(input.into()));
    }
    let multiplier = match unit.trim() {
        "" | "b" | "byte" | "bytes" => 1.0,
        "kb" => 1_000.0,
        "mb" => 1_000_000.0,
        "gb" => 1_000_000_000.0,
        "tb" => 1_000_000_000_000.0,
        "kib" => 1024.0,
        "mib" => 1024.0 * 1024.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return Err(MemoryError::InvalidBudget(input.into())),
    };
    let bytes = value * multiplier;
    if bytes > u64::MAX as f64 {
        return Err(MemoryError::InvalidBudget(input.into()));
    }
    Ok(bytes.floor() as u64)
}

fn update_max(value: &AtomicU64, candidate: u64) {
    let mut current = value.load(Ordering::Acquire);
    while candidate > current {
        match value.compare_exchange_weak(current, candidate, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

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

    fn resource_report(state: &MemoryState) -> MemoryResourceReport {
        state
            .report()
            .resources
            .into_iter()
            .find(|report| report.resource.id == "test")
            .unwrap()
    }

    fn concurrent_reservations(pool: &MemoryPool) -> Vec<MemoryLease> {
        (0..32)
            .map(|_| {
                let pool = pool.clone();
                thread::spawn(move || pool.reserve(10))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|handle| handle.join().unwrap().ok())
            .collect()
    }

    #[test]
    fn parses_absolute_and_percentage_budgets() {
        assert_eq!(
            "8 GiB".parse(),
            Ok(MemoryBudget::Bytes(8 * 1024_u64.pow(3)))
        );
        assert_eq!("70% total".parse(), Ok(MemoryBudget::PercentTotal(0.7)));
        assert_eq!(
            "60% available".parse(),
            Ok(MemoryBudget::PercentAvailable(0.6))
        );
        assert_eq!("auto".parse(), Ok(MemoryBudget::Auto));
    }

    #[test]
    fn resolves_budgets_against_the_correct_capacity() {
        let resource = resource();
        assert_eq!(MemoryBudget::Auto.resolve(&resource), Ok(400));
        assert_eq!(MemoryBudget::PercentTotal(0.5).resolve(&resource), Ok(500));
        assert_eq!(
            MemoryBudget::PercentAvailable(0.5).resolve(&resource),
            Ok(250)
        );
    }

    #[test]
    fn leases_enforce_and_release_shared_capacity() {
        let state = MemoryState::discover();
        state.register_device(resource());
        let pool = state.pool("test", MemoryBudget::Bytes(300)).unwrap();
        let lease = pool.reserve(200).unwrap();
        assert_eq!(pool.remaining(), 100);
        assert!(pool.reserve(101).is_err());
        drop(lease);
        assert_eq!(pool.remaining(), 300);
        assert_eq!(pool.high_water(), 200);
    }

    #[test]
    fn reservation_lifecycle_keeps_pool_and_resource_reports_aligned() {
        let state = MemoryState::discover();
        state.register_device(resource());
        let pool = state.pool("test", MemoryBudget::Bytes(300)).unwrap();

        let zero = pool.reserve(0).unwrap();
        assert_eq!(pool.reserved(), 0);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 0);
        drop(zero);

        let lease = pool.reserve(120).unwrap();
        let clone = lease.clone();
        let pool_report = pool.report();
        let state_report = resource_report(&state);
        assert_eq!(pool_report.reserved_bytes, 120);
        assert_eq!(pool_report.high_water_bytes, 120);
        assert_eq!(state_report.laddu_reserved_bytes, 120);
        assert_eq!(state_report.laddu_high_water_bytes, 120);

        drop(lease);
        assert_eq!(pool.reserved(), 120);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 120);
        drop(clone);

        let pool_report = pool.report();
        let state_report = resource_report(&state);
        assert_eq!(pool_report.reserved_bytes, 0);
        assert_eq!(pool_report.high_water_bytes, 120);
        assert_eq!(state_report.laddu_reserved_bytes, 0);
        assert_eq!(state_report.laddu_high_water_bytes, 120);
    }

    #[test]
    fn multiple_pools_share_physical_capacity_without_failed_side_effects() {
        let state = MemoryState::discover();
        state.register_device(resource());
        let first = state.pool("test", MemoryBudget::Bytes(400)).unwrap();
        let second = state.pool("test", MemoryBudget::Bytes(400)).unwrap();

        let first_lease = first.reserve(300).unwrap();
        assert!(second.reserve(201).is_err());
        assert_eq!(second.report().reserved_bytes, 0);
        assert_eq!(second.report().high_water_bytes, 0);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 300);

        let second_lease = second.reserve(200).unwrap();
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 500);
        assert_eq!(resource_report(&state).laddu_high_water_bytes, 500);
        drop(first_lease);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 200);
        drop(second_lease);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 0);
    }

    #[test]
    fn concurrent_reservations_are_atomic_with_and_without_state() {
        let state = MemoryState::discover();
        state.register_device(resource());
        let attached = state.pool("test", MemoryBudget::Bytes(100)).unwrap();
        let leases = concurrent_reservations(&attached);
        assert_eq!(leases.len(), 10);
        assert_eq!(attached.reserved(), 100);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 100);
        drop(leases);
        assert_eq!(attached.reserved(), 0);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 0);

        let detached = state.pool("test", MemoryBudget::Bytes(100)).unwrap();
        drop(state);
        let leases = concurrent_reservations(&detached);
        assert_eq!(leases.len(), 10);
        assert_eq!(detached.reserved(), 100);
        assert_eq!(detached.high_water(), 100);
        drop(leases);
        assert_eq!(detached.reserved(), 0);
    }

    #[test]
    fn overflow_failure_leaves_reservation_counters_unchanged() {
        let state = MemoryState::discover();
        let mut maximum = resource();
        maximum.total_bytes = Some(u64::MAX);
        maximum.available_bytes = Some(u64::MAX);
        state.register_device(maximum);
        let pool = state.pool("test", MemoryBudget::Bytes(u64::MAX)).unwrap();

        let lease = pool.reserve(u64::MAX).unwrap();
        assert!(pool.reserve(1).is_err());
        assert_eq!(pool.reserved(), u64::MAX);
        assert_eq!(pool.high_water(), u64::MAX);
        let report = resource_report(&state);
        assert_eq!(report.laddu_reserved_bytes, u64::MAX);
        assert_eq!(report.laddu_high_water_bytes, u64::MAX);
        drop(lease);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(resource_report(&state).laddu_reserved_bytes, 0);
    }

    #[test]
    fn decisions_fit_the_largest_safe_chunk() {
        let decision = MemoryDecision::fit("test", 100, 8, 1_000, 1_000, "streaming").unwrap();
        assert_eq!(decision.chunk_events, 112);
        assert_eq!(decision.estimated_peak_bytes, 996);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn reports_current_process_memory() {
        let state = MemoryState::discover();
        let first = state.report().process.unwrap();
        let second = state.report().process.unwrap();
        assert!(first.resident_bytes > 0);
        assert!(first.virtual_bytes >= first.resident_bytes);
        assert!(second.sampled_high_water_bytes >= first.resident_bytes);
        assert!(second.sampled_high_water_bytes >= second.resident_bytes);
    }
}
