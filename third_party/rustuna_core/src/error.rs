use std::panic::Location;

/// Error type used by `rustuna_core`.
///
/// The error stores a coarse-grained [`ErrorKind`], an optional human-readable reason, and the
/// call site where it was constructed.
#[derive(Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub reason: String,
    pub location: &'static Location<'static>,
}
impl Error {
    /// Creates a new error with the given kind.
    #[track_caller]
    pub fn new(kind: ErrorKind) -> Self {
        Self::with_reason(kind, String::new())
    }

    /// Creates a new error with the given kind and reason.
    #[track_caller]
    pub fn with_reason<T: Into<String>>(kind: ErrorKind, reason: T) -> Self {
        Self {
            kind,
            reason: reason.into(),
            location: Location::caller(),
        }
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.reason.is_empty() {
            write!(f, "{:?}", self.kind)?;
        } else {
            write!(f, "{:?}: {}", self.kind, self.reason)?;
        }
        write!(f, " (at {}:{})", self.location.file(), self.location.line())?;
        Ok(())
    }
}

/// Enumerates high-level error categories returned by `rustuna_core`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorKind {
    ObjectiveError,
    SamplerError,
    StorageError,
    DuplicatedStudy,
    StudyNotFound,
    TrialNotFound,
    TrialDiscarded,
    AttrNotFound,
    TrialQueueEmpty,
    AttrOverwriteNotAllowed,
    InvalidObjectiveValues,
    TrialAlreadyFinished,
    UnsupportedSearchSpace,
    UnsupportedMultiObjective,
    NoCompletedTrial,
    IncompatibleDistribution,
    InvalidFixedParam,
    /// A required optional dependency (e.g. the `cmaes` PyPI package) is not installed.
    /// The Python bindings map this kind to Python's `ImportError`.
    MissingDependency,
    Unexpected,
    ImportanceEvaluatorError,
}
