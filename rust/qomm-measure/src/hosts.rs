//! Stable labels for the machines the measurements were taken on.
//!
//! The reasoning is `mvp/qomm/scripts/hosts.py`'s, and so is the table: real
//! host names identify people and networks, so the label is applied where the
//! name is recorded rather than scrubbed before publication --- a repository
//! that is only safe to publish if someone remembers a step is not safe.
//!
//! What used to be here was a second copy of that table, written out in full.
//! It was held to the Python one by a test, which kept the two honest with each
//! other and did nothing at all about the thing that mattered: both files ship,
//! so both published the names. There is now one table, in
//! `scripts/host_map.txt`, which does not ship, and this reads it at run time.
//!
//! Absent, it labels nothing and every machine keeps its own name. That is the
//! normal case for anybody who is not us, and it is the right answer for them.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// Where the table is, if there is one: `QOMM_HOST_MAP`, else the copy beside
/// the Python that reads the same file.
fn map_path() -> PathBuf {
    if let Ok(path) = std::env::var("QOMM_HOST_MAP") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/host_map.txt")
}

/// Every machine the local table names, or none.
pub fn labels() -> BTreeMap<String, String> {
    labels_from(&map_path())
}

/// The same, from a named file. Split out so the tests can exercise the
/// parsing and the lookup without an environment variable between them ---
/// four tests sharing one variable is a race, and it was one.
pub fn labels_from(path: &std::path::Path) -> BTreeMap<String, String> {
    let mut table = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return table;
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        if let (Some(name), Some(published)) = (parts.next(), parts.next()) {
            table.insert(name.to_string(), published.to_string());
        }
    }
    table
}

/// The published name for a machine. Unknown machines keep their name.
pub fn label(node: &str) -> String {
    lookup(&labels(), node)
}

/// The lookup itself: exact first, then without a domain, then unchanged.
pub fn lookup(table: &BTreeMap<String, String>, node: &str) -> String {
    let short = node.split('.').next().unwrap_or(node);
    table
        .get(node)
        .or_else(|| table.get(short))
        .cloned()
        .unwrap_or_else(|| node.to_string())
}

/// The label for the machine currently running, for harnesses to record.
///
/// `QOMM_HOST` wins when set, so a run inside a container --- where the node
/// name is a hash that no table can hold --- can still say where it really was.
pub fn this_host() -> String {
    if let Ok(name) = std::env::var("QOMM_HOST") {
        if !name.is_empty() {
            return label(&name);
        }
    }
    label(&node_name())
}

fn node_name() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// A table of machines that do not exist, so the test says what it means
    /// about the lookup without saying anything about anybody's network.
    ///
    /// Written with `std` rather than `tempfile`: this crate has no
    /// dependencies, which is worth more than the few lines that would save.
    fn fixture(tag: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("qomm-host-map-{}-{}.txt", std::process::id(), tag));
        std::fs::write(
            &path,
            concat!(
                "# a comment, and a blank line follows\n",
                "\n",
                "grinder      site-one   # trailing comment\n",
                "kettle.local site-two\n"
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn a_listed_machine_gets_its_label() {
        let path = fixture("listed");
        let table = labels_from(&path);
        assert_eq!(lookup(&table, "grinder"), "site-one");
        // A domain is stripped before the second look, so a machine listed
        // without one is found however it announces itself.
        assert_eq!(lookup(&table, "grinder.internal"), "site-one");
        // One listed *with* `.local` is not in the table under its bare form
        // and is left alone --- which is what the Python does, and being the
        // same as the Python is the point.
        assert_eq!(lookup(&table, "kettle.local"), "site-two");
        assert_eq!(lookup(&table, "kettle"), "kettle");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unknown_machine_keeps_its_name() {
        let path = fixture("unknown");
        let table = labels_from(&path);
        assert_eq!(
            lookup(&table, "somebody-elses-laptop"),
            "somebody-elses-laptop"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn no_table_means_no_labelling_rather_than_a_failure() {
        let table = labels_from(Path::new("/nonexistent/host_map.txt"));
        assert!(table.is_empty());
        assert_eq!(lookup(&table, "grinder"), "grinder");
    }

    #[test]
    fn the_environment_can_say_where_a_container_really_is() {
        // The only test here that touches the environment, so nothing races
        // with it over the same two variables.
        let path = fixture("container");
        unsafe {
            std::env::set_var("QOMM_HOST_MAP", &path);
            std::env::set_var("QOMM_HOST", "grinder");
        }
        assert_eq!(this_host(), "site-one");
        unsafe {
            std::env::remove_var("QOMM_HOST");
            std::env::remove_var("QOMM_HOST_MAP");
        }
        let _ = std::fs::remove_file(&path);
    }
}
