; memex-owned supplement to the vendored upstream tags queries. Board #1038.
;
; WHY THIS FILE EXISTS. The upstream TypeScript tags.scm is 573 bytes and is a SUPPLEMENT to the
; JavaScript query (TS inherits the JS grammar), not a standalone one. Measured: run alone against
; a React/TSX sample it returns ZERO definitions. So .ts/.tsx must be JS-tags + TS-tags + this.
;
; Everything here was demanded by the round-2 council with a measured justification. Do not add a
; pattern to this file without measuring it on a real repo first - the naive version of the
; wrapper rule below produced 405 garbage matches.

; --- React/TS wrapper components (extractor lens) --------------------------------------------
; `const Button = forwardRef((props, ref) => ...)` is a component definition, but upstream's
; variable_declarator rule only matches a DIRECT (arrow_function)/(function_expression) value, so
; a wrapped one is invisible. Measured: 27 forwardRef components in smokesignals-web are lost
; without this.
;
; The allowlist is LOAD-BEARING. Matching any call_expression whose argument is an arrow function
; produced 405 matches on the same repo - every useEffect, every .map(), every promise callback
; becoming a fake "definition". Keep this list explicit and short; widen it only with a measurement.
(variable_declarator
  name: (identifier) @name
  value: (call_expression
           function: (identifier) @_wrapper
           (#match? @_wrapper "^(forwardRef|memo|lazy|observer)$"))) @definition.function

; Same, for the namespaced form `React.memo(...)` / `React.forwardRef(...)`.
(variable_declarator
  name: (identifier) @name
  value: (call_expression
           function: (member_expression
                       property: (property_identifier) @_wrapper)
           (#match? @_wrapper "^(forwardRef|memo|lazy)$"))) @definition.function

; --- TypeScript declarations upstream's tags.scm omits (prior-art lens) -----------------------
; type aliases and enums are first-class definitions in a TS codebase and neither appears in the
; upstream query.
(type_alias_declaration
  name: (type_identifier) @name) @definition.type

(enum_declaration
  name: (identifier) @name) @definition.enum

; --- modern namespace syntax (adversary lens) -------------------------------------------------
; The upstream TS query matches (module name: (identifier)) which is the LEGACY `module X {}`
; form. Modern `namespace X {}` parses as internal_module, so upstream is stale against its own
; grammar and silently misses every namespace in the codebase.
(internal_module
  name: (identifier) @name) @definition.module
