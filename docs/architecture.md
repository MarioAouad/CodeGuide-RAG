# Implementation notes

CodeGuide keeps the first version in a few Python modules. There is no service
layer, database, agent framework, or background worker to operate.

## Data contracts

`Source` represents one nonempty notebook cell or one complete plain file.
Notebook-only fields are `None` for plain files. `file_hash` is SHA-256 over the
original file bytes, including the notebook JSON; saved outputs never become
source text even though changing them changes the file hash.

`Chunk` contains the plan's required fields:

| Field | Meaning |
| --- | --- |
| `chunk_id` | SHA-256 of file identity, cell position, and original character span |
| `file_name`, `file_hash` | Display basename and original content identity |
| `cell_index` | One-based original notebook cell position, including other cell types |
| `cell_id`, `cell_type` | Original cell identity and `code`/`markdown` type |
| `heading` | Latest preceding Markdown heading, retained in full |
| `start_line`, `end_line` | One-based inclusive cell-relative or file-relative lines |
| `text` | Exact original substring, including its original whitespace |
| `embedding_text` | Context prefix plus the original substring |
| `start_offset`, `end_offset` | Zero-based, end-exclusive character positions in the source |
| `function_name` | Top-level Python definition name or names when available |

`Hit` associates a chunk with its cosine similarity. Citation labels are assigned
to the ordered hits for each question, rather than stored permanently. This keeps
`[S1]` aligned with the first displayed source card.

## State and failure behavior

The Streamlit session owns the active `VectorIndex`, selection state, and latest
answer. Only the CPU embedding model is shared through `st.cache_resource`.
Files selected in the uploader remain in memory; the app writes extracted
excerpts and vectors to its application-controlled cache paths.

Indexing first validates the complete selection. It then loads a matching cache
or builds a fresh candidate. The session adopts the candidate only after all
steps succeed. A malformed replacement therefore cannot destroy the active
in-memory index. Changed selections disable question submission and clear the
displayed answer until indexing succeeds.

Each persisted file is replaced atomically and the manifest is published last.
A process interrupted between file replacements can leave an invalid cache;
checksums detect that case on the next load and trigger rebuilding. This is
appropriate for a local demo; it is not a transactional multi-user storage design.

Generation failures leave retrieval results and the index available. Known SDK
errors become sanitized messages for missing credentials, invalid requests,
permissions, unavailable models, exhausted quotas, timeouts, and connection
failures. The UI also catches unexpected adapter/model errors without displaying
raw exception strings that might contain sensitive data.

## Trust boundaries

Source files are data. JSON decoding and AST parsing never execute user code.
Notebook links, saved outputs, and attachments are ignored. Source filenames are
reduced to display basenames, and corpus hashes determine cache directories.
Duplicate basenames are rejected to keep references unambiguous.

The generation adapter separates a fixed system instruction from a JSON payload
containing the question and evidence. Source instructions are explicitly marked
untrusted. No Gemini tools are configured, and automatic function calling is
disabled. API credentials remain in local settings and are absent from prompts,
chunk records, manifests, and object representations.

The model can still produce unsupported explanations or follow malicious source
instructions. Label validation detects unknown or absent citations, not semantic
truth. Human review of the source cards and a live injection test remain part of
acceptance. Plain-text answer rendering prevents model text from loading remote
images or turning HTML into interface elements.

## Deliberate limits

- Chunk windows use the embedding tokenizer, avoiding silent source truncation.
  Very long single lines may be fragmented without overlap; offsets still retain
  every non-whitespace character and identify the exact excerpt.
- Context expansion adds adjacent cells of the opposite type. It is a small
  notebook heuristic and can omit a relevant same-type neighbor.
- Exact definition names receive a small ranking preference. There is no second
  retrieval engine or semantic relevance threshold presented as confidence.
- Cache identity includes model name and chunk settings, not the contents of an
  upstream model repository. Delete saved indexes after replacing model weights
  under the same name.
- Older cache directories retain source excerpts. Clearing a session does not
  erase disk history; the README describes removal.
- Billing tier and account-specific model access cannot be verified without the
  user's configured project. The app cannot guarantee zero billing if that
  project has paid billing enabled.

If this becomes a shared or deployed application, add authentication, isolated
storage, request limits, monitoring, stronger grounding evaluation, and a suitable
privacy policy before exposing it beyond localhost.
