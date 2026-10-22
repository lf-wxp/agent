//! Retrieval: chunking text, embedding it, and searching it — once, in memory
//! ([`search::vector_search`]), or persistently
//! ([`store::VectorStore`], behind the `sqlite` feature).

pub mod chunk;
pub mod embed;
pub mod search;
#[cfg(feature = "sqlite")]
pub mod store;
#[cfg(test)]
pub(crate) mod testing;
