//! The semantic type registry — where `CREATE SEMANTIC TYPE` definitions live
//! for the session. Both the DDL dispatch (via [`SemcastRuntime`]) and the
//! marker UDFs' `return_field_from_args` reach it through a shared `Arc`.
//!
//! [`SemcastRuntime`]: crate::index::registry::SemcastRuntime

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::error::DataFusionError;

use super::SemanticType;

/// Registered semantic types, keyed by lowercased name. Names are matched
/// case-insensitively (a `CAST(x AS MeetingFacts)` resolves `meetingfacts`).
#[derive(Debug, Default)]
pub struct TypeRegistry {
    types: Mutex<HashMap<String, Arc<SemanticType>>>,
}

impl TypeRegistry {
    /// Register a type. Re-registering an existing name is an error — there is
    /// no `CREATE OR REPLACE` yet, so a name is defined once per session.
    pub fn register(&self, ty: SemanticType) -> crate::Result<()> {
        let key = ty.name.to_lowercase();
        let mut types = self.types.lock().expect("type registry poisoned");
        if types.contains_key(&key) {
            return Err(crate::SemcastError::DataFusion(DataFusionError::Plan(
                format!(
                    "semantic type {} is already defined; CREATE OR REPLACE is not \
                     implemented yet, so use a new name",
                    ty.name
                ),
            )));
        }
        types.insert(key, Arc::new(ty));
        Ok(())
    }

    /// The type registered under `name` (case-insensitive), if any.
    pub fn get(&self, name: &str) -> Option<Arc<SemanticType>> {
        self.types
            .lock()
            .expect("type registry poisoned")
            .get(&name.to_lowercase())
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty(name: &str) -> SemanticType {
        SemanticType {
            name: name.to_owned(),
            version: "v1".to_owned(),
            fields: vec![],
            together: vec![],
        }
    }

    #[test]
    fn register_then_get_is_case_insensitive() {
        let registry = TypeRegistry::default();
        registry.register(ty("MeetingFacts")).unwrap();
        assert!(registry.get("meetingfacts").is_some());
        assert!(registry.get("MEETINGFACTS").is_some());
        assert!(registry.get("Other").is_none());
    }

    #[test]
    fn duplicate_registration_errors() {
        let registry = TypeRegistry::default();
        registry.register(ty("Facts")).unwrap();
        let err = registry.register(ty("facts")).unwrap_err();
        assert!(err.to_string().contains("already defined"), "got: {err}");
    }
}
