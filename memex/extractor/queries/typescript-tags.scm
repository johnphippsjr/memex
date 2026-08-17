; VENDORED - do not edit by hand. Board #1038.
; source: tree-sitter/tree-sitter-typescript queries/tags.scm
; package: tree-sitter-typescript==0.23.2
; Vendored deliberately rather than importing the pip package at runtime: three
; independently-versioned grammar sources (tslp's grammar + these two wheels) is a
; slow leak. We compile this text against tslp's grammars. If you bump it, RE-MEASURE
; the fixture counts - do not assume.

(function_signature
  name: (identifier) @name) @definition.function

(method_signature
  name: (property_identifier) @name) @definition.method

(abstract_method_signature
  name: (property_identifier) @name) @definition.method

(abstract_class_declaration
  name: (type_identifier) @name) @definition.class

(module
  name: (identifier) @name) @definition.module

(interface_declaration
  name: (type_identifier) @name) @definition.interface

(type_annotation
  (type_identifier) @name) @reference.type

(new_expression
  constructor: (identifier) @name) @reference.class
