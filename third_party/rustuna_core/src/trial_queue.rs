use std::collections::VecDeque;

use crate::{Error, ErrorKind, Result};

/// Queue abstraction used by [`crate::study::Study::enqueue_trial`].
///
/// Rustuna keeps queuing outside the storage layer so that users who do not use queued trials do
/// not pay coordination overhead on every trial creation.
pub trait TrialQueue: Send + Sync {
    /// Pushes a trial ID into the queue.
    fn enqueue(&mut self, trial_id: u32) -> Result<()>;
    /// Pops the next queued trial ID.
    fn dequeue(&mut self) -> Result<u32>;
}

/// In-memory queue implementation used by default.
#[derive(Default)]
pub struct InMemoryTrialQueue {
    queue: VecDeque<u32>,
}

impl InMemoryTrialQueue {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl TrialQueue for InMemoryTrialQueue {
    fn enqueue(&mut self, trial_id: u32) -> Result<()> {
        self.queue.push_back(trial_id);
        Ok(())
    }

    fn dequeue(&mut self) -> Result<u32> {
        self.queue
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::TrialQueueEmpty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_and_dequeue() {
        let mut queue = InMemoryTrialQueue::new();

        assert!(queue.enqueue(1).is_ok());
        assert!(queue.enqueue(2).is_ok());
        assert!(queue.enqueue(3).is_ok());

        assert_eq!(queue.dequeue().unwrap(), 1);
        assert_eq!(queue.dequeue().unwrap(), 2);
        assert_eq!(queue.dequeue().unwrap(), 3);
    }

    #[test]
    fn test_dequeue_empty_queue() {
        let mut queue = InMemoryTrialQueue::new();
        assert!(queue.dequeue().is_err());
    }
}
