## rustuna_core

### Example

```rust
use std::sync::{Arc, Mutex};

use rustuna_core::objective_single::get_best_trial;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{Direction, create_study};
use rustuna_core::Result;
use rustuna_sampler::tpe::TpeSampler;

fn main() -> Result<()> {
    let storage = InMemoryStorage::new();
    let mut study = create_study("simple-quadratic", storage, vec![Direction::Minimize])?;

    let sampler = Arc::new(Mutex::new(TpeSampler::new()));
    study.optimize(
        |mut t| {
            let x = t.suggest_float("x", 0.0, 10.0)?;
            let y = t.suggest_float("y", 0.0, 10.0)?;
            let value = (x - 3.0).powi(2) + (y - 5.0).powi(2);
            println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
            Ok(vec![value])
        },
        sampler,
        100,
    )?;

    println!("Best trial: {:?}", get_best_trial(&study)?);
    Ok(())
}
```

### Contributing

#### Build from Source

```
$ cargo build -p rustuna_core
```
