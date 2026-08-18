# Graph Report - medAssistVoDc  (2026-08-18)

## Corpus Check
- 151 files · ~66,621 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1295 nodes · 2852 edges · 62 communities (59 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e586fe1d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AppError
- pipeline.py
- api.ts
- monitoring.rs
- dependencies
- parser.py
- useChat.ts
- test_algorithm_modes.py
- calculators.rs
- devDependencies
- MonitoringPage.tsx
- admin.rs
- cn
- auth.rs
- auth.rs
- useAuth
- compilerOptions
- algorithms.rs
- exporter.py
- CalculatorCard.tsx
- RecommendationCard.tsx
- AdminUsersPage.tsx
- OllamaClient
- Быстрый старт (dev, локально)
- compilerOptions
- components.json
- schemas.py
- relay_and_collect_with_events
- stream
- toast.tsx
- vector_store.py
- get_recommendations
- OllamaEmbeddings
- PythonClient
- useToast
- run_one
- FastAPI
- router_clinical_recs.py
- ingest_document
- match_services
- test_pdf_exporter.py
- recover_delegated_user_migration
- tsconfig.json
- ingest_attachment
- AdminRoute.tsx
- ErrorBoundary
- React + TypeScript + Vite
- Settings
- auth-session.spec.ts
- medassistant-ai

## God Nodes (most connected - your core abstractions)
1. `AppError` - 97 edges
2. `AppState` - 69 edges
3. `cn()` - 44 edges
4. `apiFetch()` - 34 edges
5. `Button()` - 27 edges
6. `useAuth()` - 25 edges
7. `compilerOptions` - 22 edges
8. `compilerOptions` - 18 edges
9. `run_one()` - 18 edges
10. `summary()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `_read_source()` --calls--> `extract_text()`  [INFERRED]
  tools/algorithm_smoke.py → ai-service/src/pdf/extractor.py
- `run_one()` --calls--> `extract_text()`  [INFERRED]
  tools/minzdrav_regression.py → ai-service/src/pdf/extractor.py
- `run_one()` --calls--> `extract_diagnosis()`  [INFERRED]
  tools/minzdrav_regression.py → ai-service/src/pdf/parser.py
- `run_one()` --calls--> `parse_sections()`  [INFERRED]
  tools/minzdrav_regression.py → ai-service/src/pdf/parser.py
- `run_one()` --calls--> `build_algorithm_sections()`  [INFERRED]
  tools/minzdrav_regression.py → ai-service/src/pdf/parser.py

## Import Cycles
- None detected.

## Communities (62 total, 3 thin omitted)

### Community 0 - "AppError"
Cohesion: 0.06
Nodes (89): Config, Self, String, Vec, AppError, Error, From, Response (+81 more)

### Community 1 - "pipeline.py"
Cohesion: 0.05
Nodes (86): algorithm_user_prompt(), expand_section_prompt(), outline_user_prompt(), physician_algorithm_user_prompt(), physician_outline_user_prompt(), physician_section_prompt(), structured_algorithm_user_prompt(), structured_outline_user_prompt() (+78 more)

### Community 2 - "api.ts"
Cohesion: 0.05
Nodes (50): AdminUsersPage, App(), CalculatorGroupPage, CalculatorsPage, ChangePasswordPage, ChatPage, ClinicalRecsPage, HomePage (+42 more)

### Community 3 - "monitoring.rs"
Cohesion: 0.11
Nodes (47): ActionCount, AuditLogEntry, action_counts(), ActionCount, audit_logs(), AuditLogEntry, collect_system_snapshot(), CurrentGenerationStats (+39 more)

### Community 4 - "dependencies"
Cohesion: 0.04
Nodes (48): @base-ui/react, class-variance-authority, clsx, @fontsource-variable/geist, dependencies, @base-ui/react, class-variance-authority, clsx (+40 more)

### Community 5 - "parser.py"
Cohesion: 0.09
Nodes (45): build_algorithm_sections(), _build_diagnosis_fallback_context(), _clean_section_title(), _consume_wrapped_title(), _diagnosis_candidate_from_filename(), _diagnosis_supported_by_text(), extract_definition(), _extract_definition_excerpt() (+37 more)

### Community 6 - "useChat.ts"
Cohesion: 0.09
Nodes (31): ChatWindow(), Props, AttachmentList(), formatFileSize(), MessageBubble(), Props, relativeTime(), Props (+23 more)

### Community 7 - "test_algorithm_modes.py"
Cohesion: 0.06
Nodes (19): generate_algorithm(), Run a synchronous generator in a thread, yielding tokens as they arrive., Stream algorithm generation from full document text. Stateless — no DB access., _stream_from_sync(), AlgorithmGenerate, algorithm_source(), FakeLlm, test_algorithm_request_accepts_supported_modes_and_rejects_unknown_mode() (+11 more)

### Community 8 - "calculators.rs"
Cohesion: 0.15
Nodes (38): b(), bad(), calculate(), CalculatorField, CalculatorGroup, CalculatorMetadata, CalculatorRegistry, CalculatorResult (+30 more)

### Community 9 - "devDependencies"
Cohesion: 0.05
Nodes (39): eslint, @eslint/js, eslint-plugin-react-hooks, eslint-plugin-react-refresh, devDependencies, eslint, @eslint/js, eslint-plugin-react-hooks (+31 more)

### Community 10 - "MonitoringPage.tsx"
Cohesion: 0.09
Nodes (29): ActionCount, AuditLogEntry, CurrentGenerationStats, GpuMetric, MonitoringSummary, SystemMetricSample, SystemSnapshot, useMonitoring() (+21 more)

### Community 11 - "admin.rs"
Cohesion: 0.17
Nodes (33): AdminUserSummary, Algorithm, ChatMessage, ChatSession, Document, NaiveDateTime, Option, String (+25 more)

### Community 12 - "cn"
Cohesion: 0.14
Nodes (22): baseNavItems, Button(), buttonVariants, Checkbox(), ConfirmDialogProps, Dialog(), DialogContent(), DialogDescription() (+14 more)

### Community 13 - "auth.rs"
Cohesion: 0.13
Nodes (22): AdminUser, AuthUser, bearer_token(), cookie_token(), ensure_bootstrap_admin(), generate_onboarding_token(), generate_temporary_password(), generate_token() (+14 more)

### Community 14 - "auth.rs"
Cohesion: 0.20
Nodes (31): active_sessions(), AuthUserResponse, change_password(), ChangePasswordRequest, clean_optional(), clear_cookie_expires_same_scope(), clear_session_cookie_headers(), complete_onboarding() (+23 more)

### Community 15 - "useAuth"
Cohesion: 0.17
Nodes (20): ALGORITHM_MODES, Props, fields, ProfileForm(), ProfileFormProps, Card(), CardAction(), CardContent() (+12 more)

### Community 16 - "compilerOptions"
Cohesion: 0.07
Nodes (29): compilerOptions, allowImportingTsExtensions, baseUrl, erasableSyntaxOnly, jsx, lib, module, moduleDetection (+21 more)

### Community 17 - "algorithms.rs"
Cohesion: 0.14
Nodes (22): Algorithm, AlgorithmGenerationMode, AlgorithmGenerationMode, emit_algorithm_changed(), export_pdf(), ExportPdfRequest, generate(), GenerateRequest (+14 more)

### Community 18 - "exporter.py"
Cohesion: 0.17
Nodes (27): export_pdf(), _algorithm_diagnosis(), algorithm_pdf_filename(), algorithm_pdf_title(), _append_element(), _build_styles(), _callout(), _column_widths() (+19 more)

### Community 19 - "CalculatorCard.tsx"
Cohesion: 0.13
Nodes (21): CalculatorCard(), getDefaultValues(), parseValues(), Props, config, validateFields(), Select(), SelectOption (+13 more)

### Community 20 - "RecommendationCard.tsx"
Cohesion: 0.15
Nodes (19): Props, RecommendationCard(), DocumentCard(), Props, statusConfig, DocumentList(), Props, Badge() (+11 more)

### Community 21 - "AdminUsersPage.tsx"
Cohesion: 0.13
Nodes (18): AdminUser, ResetPasswordResponse, UpdateUserPayload, useAdminUsers(), useCreateUser(), useDeleteUser(), useResetUserPassword(), UserWithTemporaryPassword (+10 more)

### Community 22 - "OllamaClient"
Cohesion: 0.13
Nodes (15): OllamaClient, get_services_df(), get_top_services(), _load(), Return top_k most relevant services by TF-IDF similarity., _get_llm(), match_services(), _parse_response() (+7 more)

### Community 23 - "Быстрый старт (dev, локально)"
Cohesion: 0.08
Nodes (23): 1. Войти в dev shell, 2. Скачать LLM-модель, 3. Запустить Ollama (если не systemd), 4. Настроить AI Service (Python), 5. Запустить AI Service, 6. Запустить Backend (Rust), 7. Запустить Frontend (React), AI Service (prefix `MED_`) (+15 more)

### Community 24 - "compilerOptions"
Cohesion: 0.09
Nodes (22): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, lib, module, moduleDetection, moduleResolution, noEmit (+14 more)

### Community 25 - "components.json"
Cohesion: 0.09
Nodes (21): aliases, components, hooks, lib, ui, utils, iconLibrary, menuAccent (+13 more)

### Community 26 - "schemas.py"
Cohesion: 0.18
Nodes (19): match(), BMIRequest, BSARequest, CalculatorResult, ChatAttachmentContext, ChatMessage, CreatinineRequest, DeleteDocumentRequest (+11 more)

### Community 27 - "relay_and_collect_with_events"
Cohesion: 0.19
Nodes (19): format_sse_event(), GenerationEventRow, insert_generation_event(), is_error_event(), parse_sse_error_message(), parse_sse_string(), relay_and_collect_with_events(), RelayResult (+11 more)

### Community 28 - "stream"
Cohesion: 0.20
Nodes (15): AccountEventRow, emit_account_event(), emit_admin_users_changed(), EventsQuery, format_event(), AuthUser, Bytes, Option (+7 more)

### Community 29 - "toast.tsx"
Cohesion: 0.15
Nodes (11): ThemeToggle(), Toast, ToastAction, ToastContext, ToastContextValue, ToastProvider(), ToastType, queryClient (+3 more)

### Community 30 - "vector_store.py"
Cohesion: 0.22
Nodes (13): add_chunks(), delete_document_chunks(), _get_client(), get_collection(), get_document_chunks(), _get_embedder(), query_chunks(), Return all chunks for a document ordered by ingestion chunk index. (+5 more)

### Community 31 - "get_recommendations"
Cohesion: 0.20
Nodes (9): get_recommendations(), RecommendationsQuery, AuthUser, Json, Query, Result, String, Value (+1 more)

### Community 32 - "OllamaEmbeddings"
Cohesion: 0.22
Nodes (5): OllamaEmbeddings, Thin wrapper around Ollama embedding models., Embed a batch of texts (batched to avoid overloading the API)., test_nomic_uses_retrieval_prefixes(), test_other_embedding_models_are_not_prefixed()

### Community 33 - "PythonClient"
Cohesion: 0.24
Nodes (7): HealthResponse, PythonClient, Error, Result, Self, String, Client

### Community 34 - "useToast"
Cohesion: 0.26
Nodes (10): ChatInput(), formatFileSize(), Props, Props, UploadZone(), useToast(), AppConfig, useAppConfig() (+2 more)

### Community 35 - "run_one"
Cohesion: 0.33
Nodes (12): check_text(), diagnosis_matches_title(), diagnosis_tokens(), download_pdf(), fetch_recommendations(), first_heading(), has_bad_heading(), main() (+4 more)

### Community 36 - "FastAPI"
Cohesion: 0.20
Nodes (7): chat_completion(), Stateless RAG chat: accept question + document_id + context, stream SSE answer., delete_document(), Delete document chunks from ChromaDB vector store., ChatRequest, create_app(), FastAPI

### Community 37 - "router_clinical_recs.py"
Cohesion: 0.25
Nodes (8): _filter_recommendations(), _fresh(), get_clinical_recommendations(), _paginate_recommendations(), fetch_recommendations(), Fetch clinical recommendations from Minzdrav API., test_pagination_returns_requested_slice_and_metadata(), test_search_is_case_insensitive_across_supported_fields()

### Community 38 - "ingest_document"
Cohesion: 0.22
Nodes (9): ingest_document(), UploadFile, Process a PDF: extract text, parse, chunk, embed. No local DB — returns results, extract_first_page(), extract_text(), extract_text_by_pages(), Extract full text from a PDF using PyMuPDF., Extract text from the first page only. (+1 more)

### Community 39 - "match_services"
Cohesion: 0.31
Nodes (9): match_services(), MatchRequest, MatchResponse, AuthUser, Json, Result, String, Vec (+1 more)

### Community 40 - "test_pdf_exporter.py"
Cohesion: 0.32
Nodes (4): Path, test_export_pdf_endpoint_returns_downloadable_pdf(), test_exported_pdf_does_not_contain_raw_markdown_table(), test_markdown_to_pdf_handles_physician_algorithm()

### Community 41 - "recover_delegated_user_migration"
Cohesion: 0.43
Nodes (7): init_pool(), recover_delegated_user_migration(), SqlitePool, run_migrations(), table_exists(), PoolConnection, Sqlite

### Community 42 - "tsconfig.json"
Cohesion: 0.25
Nodes (7): compilerOptions, baseUrl, paths, files, ./src/*, @/*, references

### Community 43 - "ingest_attachment"
Cohesion: 0.33
Nodes (6): ingest_attachment(), UploadFile, ChatAttachmentIngestResponse, Chunk, chunk_sections(), Split parsed sections into overlapping chunks respecting section boundaries.

### Community 44 - "AdminRoute.tsx"
Cohesion: 0.43
Nodes (4): SidebarContent(), AdminRoute(), canManageUsers(), UserRole

### Community 45 - "ErrorBoundary"
Cohesion: 0.29
Nodes (3): ErrorBoundary, Props, State

### Community 46 - "React + TypeScript + Vite"
Cohesion: 0.50
Nodes (3): Expanding the ESLint configuration, React Compiler, React + TypeScript + Vite

## Knowledge Gaps
- **202 isolated node(s):** `medassistant-ai`, `$schema`, `style`, `rsc`, `tsx` (+197 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AppError` connect `AppError` to `monitoring.rs`, `match_services`, `calculators.rs`, `admin.rs`, `auth.rs`, `auth.rs`, `algorithms.rs`, `stream`, `get_recommendations`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `AppState` connect `AppError` to `PythonClient`, `monitoring.rs`, `match_services`, `admin.rs`, `auth.rs`, `auth.rs`, `algorithms.rs`, `stream`, `get_recommendations`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `ingest_document()` connect `ingest_document` to `FastAPI`, `parser.py`, `ingest_attachment`, `schemas.py`, `vector_store.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **What connects `medassistant-ai`, `$schema`, `style` to the rest of the system?**
  _202 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AppError` be split into smaller, more focused modules?**
  _Cohesion score 0.06297029702970297 - nodes in this community are weakly interconnected._
- **Should `pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05005107252298264 - nodes in this community are weakly interconnected._
- **Should `api.ts` be split into smaller, more focused modules?**
  _Cohesion score 0.05201266395296246 - nodes in this community are weakly interconnected._