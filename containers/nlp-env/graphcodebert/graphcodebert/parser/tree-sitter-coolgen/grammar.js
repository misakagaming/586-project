/**
 * @file CoolGen grammar for tree-sitter
 * @author Turgay Aytac
 * @license AI4U
 * @see {@link https://dokumen.tips/documents/coolgen-syntax.html}
 */

/* eslint-disable arrow-parens */
/* eslint-disable camelcase */
/* eslint-disable-next-line spaced-comment */
/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PREC = {

  import_block: 9,
  export_block: 8,
  entact_block: 7,
  local_block: 6,
  main_blocks: 6,
  view_blocks: 4,
  view_body: 3,
  view_attributes: 2,
  sub_process: 2,
  procedure_section: 6,
  
  statement: 10,
  use_block: 9,
  import_list: 12,
  import_param: 7,
  export_list: 11,
  export_param: 5,

  conditional: -1,

  parenthesized_expression: 1,
  parenthesized_list_splat: 1,
  attribute: 10,
  string_content: 10,
  or: 10,
  and: 11,
  not: 12,
  compare: 13,
  bitwise_or: 14,
  bitwise_and: 15,
  xor: 16,
  shift: 17,
  plus: 18,
  times: 19,
  unary: 20,
  power: 21,
  call: 22,
};

module.exports = grammar({
  name: 'coolgen',

  extras: $ => [
    /[!\s\f\t\uFEFF\u2060\u200B]|\r?\n/,
    // $.newline,
    // $.line_continuation,
  ],

  conflicts: $ => [
  ],

  supertypes: $ => [
    $.expression,
    $.primary_expression,
  ],

  externals: $ => [
    $.note_terminator,
    $.statement_id,
    $.statement_part,
    $.block_id,
    $.block_terminator,
    $.booland_break,
    $.boolor_break,
    $.parameter_break,
    $.error_sentinel,
  ],

  inline: $ => [
  ],

  word: $ => $.identifier,

  rules: {

    module: $ => seq(
     $.module_definition,
     choice(seq('PROCEDURE', 'STATEMENTS'), seq('EXTERNAL', 'ACTION', 'BLOCK')),
     repeat($.statement),
     '+---',
    ),

    module_definition: $ => seq(
      '+->',
      field('name', $.identifier),
      optional(field('date',$.date)),
      optional(field('time',$.time)),
      optional(field('imports',$.imports_block)),
      optional(field('exports',$.exports_block)),
      optional(field('entity_actions',$.entityactions_block)),
      optional(field('locals',$.locals_block)),
    ),

    imports_block: $ => prec(PREC.import_block,seq(
      'IMPORTS',
      ':',
      repeat1($.view_block),
    )),

    exports_block: $ => prec(PREC.export_block,seq(
      'EXPORTS',
      ':',
      repeat1($.view_block),
    )),

    entityactions_block: $ => prec(PREC.entact_block,seq(
      'ENTITY',
      'ACTIONS',
      ':',
      repeat1($.view_block),
    )),

    locals_block: $ => prec(PREC.local_block,seq(
      'LOCALS',
      ':',
      repeat1($.view_block),
    )),

    view_block: $ => prec(PREC.view_blocks,choice(
      $.workview_block,
      $.entityview_block,
      $.groupview_block,
    )),

    workview_block: $ => prec(PREC.view_blocks,seq(
      'Work',
      'View',
      $.view_body,
    )),

    entityview_block: $ => prec(PREC.view_blocks,seq(
      'Entity',
      'View',
      $.view_body,
    )),

    groupview_block: $ => prec(PREC.view_blocks,seq(
      'Group',
      'View',
      optional(field('view_id', seq('(',$.integer,')'))),
      field('view_name', $.identifier),
      choice($.workview_block, $.entityview_block), 
    )),

    view_body: $ => prec(PREC.view_body,seq(
      field('view_name', $.identifier),
      optional(field('view_descriptor', $.identifier)),
      optional($.view_options),
      $.view_attributes,
    )),

    view_options: $ => seq(
      '(', 
      commaSep1(field('view_option', choice('Transient', 'Mandatory', 'Optional', seq('Export', 'only'), seq('Import', 'only')))),
      ')'
    ),

    view_attributes: $ => prec(PREC.view_attributes,
	repeat1(field('view_attribute', $.identifier)),
    ),

    statement: $ => choice(
      $.note_statement,
      $.set_statement,
      $.move_statement,
      $.use_statement,
      $.if_statement,
      $.case_statement,
      $.for_statement,
      $.while_statement,
      $.repeat_statement,
      $.escape_statement,
      $.create_statement,
      $.update_statement,
      $.delete_statement,
      $.read_statement,
      $.readeach_statement,
      $.summarize_statement,
      $.empty_statement,
      $.exitstate_statement,
    ),

    note_statement: $ => seq(
      $.statement_id,
      'NOTE',
      ':',
      $.newline,
      repeat(seq($.statement_part, $.noteline, $.newline)),
      // $.note_terminator,
    ),

    set_statement: $ => seq(
      $.statement_id,
      'SET',
      field('left', choice($.group_subscript, $.group_last, $.attribute)),
      optional(choice('ROUNDED', seq('NOT','ROUNDED'))),
      'TO',
      field('right', $.expression),
      $.newline,
    ),

    move_statement: $ => seq(
      $.statement_id,
      'MOVE',
      field('right', $.identifier),
      optional(field('source_view_descriptor', $.identifier)),
      'TO',
      field('left', $.identifier),
      optional(field('dest_view_descriptor', $.identifier)),
      $.newline,
    ),

    use_statement: $ => seq(
      $.statement_id,
      optional(repeat('*')), 
      'USE',
      field('module_name', $.identifier),
      $.newline,
      repeat($.import_param),
      repeat($.export_param),
      // $.block_terminator,
    ),

    import_param: $ => prec.left(14, seq(
      $.statement_part,
      optional(repeat('*')), 
      optional(seq('WHICH','IMPORTS',':')),
      choice('Entity','Work', 'Group'),
      'View',
      field('source_view_name', $.identifier),
      optional(field('source_view_descriptor', $.identifier)),
      'TO',
      choice('Entity','Work', 'Group'),
      'View',
      field('dest_view_name', $.identifier),
      optional(field('dest_view_descriptor', $.identifier)),
      $.newline,
    )),

    export_param: $ => prec.left(12, seq(
      $.statement_part,
      optional(repeat('*')), 
      optional(seq('WHICH','EXPORTS',':')),
      choice('Entity','Work', 'Group'),
      'View',
      field('dest_view_name', $.identifier),
      optional(field('dest_view_descriptor', $.identifier)),
      'FROM',
      choice('Entity','Work', 'Group'),
      'View',
      field('source_view_name', $.identifier),
      optional(field('source_view_descriptor', $.identifier)),
      $.newline,
    )),

    if_statement: $ => seq(
      $.statement_id,
      $.cblock,
      'IF',
      field('condition', $.expression),
      $.newline,
      repeat($.statement),
      optional(seq($.elseif_statement, repeat1($.statement))), 
      optional(seq($.else_statement, repeat1($.statement))), 
      $.block_id,
     '+--',
      $.newline,
    ),

    elseif_statement: $ => seq(
      $.block_id,
      $.csblock,
      'ELSEIF',
      $.newline,
    ),

    else_statement: $ => seq(
      $.block_id,
      $.csblock,
      'ELSE',
      $.newline,
    ),

    case_statement: $ => seq(
      $.statement_id,
      $.cblock,
      'CASE',
      'OF',
      field('condition', $.expression),
      $.newline,
      repeat1($.case_process),
      optional($.otherwise_process),
      $.block_id,
     '+--',
      $.newline,
    ),

    case_process: $ => prec.right(PREC.sub_process, seq(
      $.block_id,
      $.csblock,
      'CASE',
      $.constant,
      $.newline,
      repeat($.statement),
    )),

    otherwise_process: $ => prec.right(PREC.sub_process, seq(
      $.block_id,
      $.csblock,
      'OTHERWISE',
      $.newline,
      repeat($.statement),
    )),

    for_statement: $ => seq(
      $.statement_id,
      $.lblock,
      'FOR',
      field('left', choice($.attribute, $.group_subscript)),
      'FROM',
      $.integer,
      'TO',
      field('right', choice($.integer,$.group_last)),
      optional(seq('BY',$.integer)),
      $.newline,
      field('block', repeat($.statement)),
      $.block_id,
     '+--',
      $.newline,
    ),

    while_statement: $ => seq(
      $.statement_id,
      $.lblock,
      'WHILE',
      field('condition', $.expression),
      $.newline,
      repeat($.statement),
      $.block_id,
     '+--',
      $.newline,
    ),

    repeat_statement: $ => seq(
      $.statement_id,
      $.lblock,
      'REPEAT',
      $.newline,
      repeat($.statement),
      $.until_statement,
    ),

    until_statement: $ => seq(
      $.block_id,
     '+--',
     'UNTIL',
      field('condition', $.expression),
      $.newline,
     ),
    	    
    targeting_statement: $ => seq(
      $.statement_part,
      'TARGETING',
      field('group_view', $.identifier),
      'FROM',
      'THE',
      'BEGINNING',
      'UNTIL',
      'FULL',
      $.newline,
    ),

    escape_statement: $ => seq(
      $.statement_id,
      '<',
      repeat1('-'),
      'ESCAPE',
      $.newline,
    ),

    exitstate_statement: $ => seq(
      $.statement_id,
      'EXIT',
      'STATE',
      'IS',
      $.identifier,
      field('designation', optional(seq('WITH', 'ROLLBACK'))),
      $.newline,
    ),

    create_statement: $ => seq(
      $.statement_id,
      $.cblock,
      'CREATE',
      $.identifier,
      $.newline,
      repeat1($.initialize_statement),
      repeat1($.when_statement),
      $.block_id,
     '+--',
      $.newline,
    ),

    update_statement: $ => seq(
      $.statement_id,
      $.cblock,
      'UPDATE',
      $.identifier,
      $.newline,
      repeat1($.initialize_statement),
      repeat1($.when_statement),
      $.block_id,
     '+--',
      $.newline,
    ),

    delete_statement: $ => seq(
      $.statement_id,
      'DELETE',
      $.identifier,
      $.newline,
    ),

    initialize_statement: $ => seq(
      $.statement_id,
      'SET',
      $.identifier,
      'TO',
      $.expression,
      $.newline,
    ),

    read_statement: $ => seq(
      $.statement_id,
      $.lblock,
      'READ',
      optional($.query_directives),
      commaSep1(field('view_name',$.identifier)),
      $.newline,
      repeat($.sort_statement),
      $.where_statement,
      repeat1($.when_statement),
      $.block_id,
     '+--',
      $.newline,
    ),

    readeach_statement: $ => seq(
      $.statement_id,
      $.lblock,
      'READ',
      'EACH',
      optional($.query_directives),
      commaSep1(field('view_name',$.identifier)),
      $.newline,
      repeat($.sort_statement),
      $.where_statement,
      repeat1($.statement),
      $.block_id,
     '+--',
      $.newline,
    ),

    summarize_statement: $ => seq(
      $.statement_id,
      $.lblock,
      'SUMMARIZE',
      optional($.query_directives),
      commaSep1(field('view_name',$.identifier)),
      $.newline,
      repeat1($.placing_statement),
      $.where_statement,
      repeat1($.when_statement),
      $.block_id,
     '+--',
      $.newline,
    ),

    placing_statement: $ => seq(
      $.statement_part,
      optional('PLACING'),
      $.query_basic_call,
      'INTO',
      $.attribute,
      $.newline,
    ),

    query_directives: $ => seq(
      '(',
      commaSep1(choice(seq('Isolation', 'UR'), seq('Select', 'Only'), seq('Optimize', '1'), seq('Cursor', 'Only'))),
      ')',
    ),

    where_statement: $ => seq(
      $.statement_part,
      'WHERE',
      field('query_condition', $.query_condition),
      $.newline,
    ),

    sort_statement: $ => seq(
      $.statement_part,
      optional('AND'),
      'SORTED',
      'BY',
      optional(choice('ASCENDING', 'DESCENDING')),
      $.entity_attribute,
      $.newline,
    ),

    when_statement: $ => seq(
      $.block_id,
      $.csblock,
      'WHEN',
      field('result', $.query_result),
      $.newline,
      repeat1($.statement),
    ),

    query_result: $ => choice(
      'successful',
      seq('already', 'exists'),
      seq('permitted', 'value', 'violation'),
      seq('not', 'found'),
      seq('not', 'unique'),
      seq('permitted', 'value', 'violation'),
    ),

    empty_statement: $ => seq(
      $.statement_id,
      $.newline,
    ),

    // identifier: _ => /[_\p{XID_Start}][_\p{XID_Continue}]*/,
    identifier: _ => token(seq(
       /[0-9A-Za-z]/,
       /[0-9A-Za-z_]*/,
       /[A-Za-z]+/,
       /[0-9A-Za-z_]*/,
    )),

    date: _ => /[0-9]+\/[0-9]+\/[0-9]+/,

    time: _ => /[0-9]+[:-][0-9]+/,

    expression: $ => choice(
      $.comparison_operator,
      $.not_operator,
      $.boolean_operator,
      $.primary_expression,
    ),

    comparison_operator: $ => prec.left(PREC.compare, seq(
      $.primary_expression,
      repeat1(seq(
        field('operators',
          choice(
            '<',
            '<=',
            '=',
            '^=',
            '>=',
            '>',
            '<>',
            'IN',
            alias(seq('not', 'in'), 'not in'),
          )),
        $.primary_expression,
      )),
    )),

    primary_expression: $ => choice(
      $.string,
      $.integer,
      $.float,
      $.true,
      $.false,
      $.spaces,
      $.timestamp,
      $.userid,
      $.binary_operator,
      $.attribute,
      $.group_subscript,
      $.group_last,
      $.unary_operator,
      $.subscript,
      $.parenthesized_expression,
      $.basic_call,
    ),

    attribute: $ => seq(
      field('view_name', $.identifier),      
      optional(field('view_descriptor', $.identifier)),
      field('view_attribute', $.identifier),
    ),

    entity_attribute: $ => seq(
      field('view_name', $.identifier),      
      field('view_attribute', $.identifier),
    ),

    group_subscript: $ => seq(
      'SUBSCRIPT',
      'OF',
      field('view_name', $.identifier),      
    ),

    group_last: $ => seq(
      'LAST',
      'OF',
      field('view_name', $.identifier),      
    ),

    constant: $ => choice(
      $.string,
      $.integer,
      $.float,
      $.true,
      $.false,
    ),

    not_operator: $ => prec(PREC.not, seq(
      'NOT',
      field('argument', $.expression),
    )),

    boolean_operator: $ => choice(
      prec.left(PREC.and, seq(
        field('left', $.expression),
        $.booland_break,
        // field('operator', 'AND'),
        field('right', $.expression),
      )),
      prec.left(PREC.or, seq(
        field('left', $.expression),
        $.boolor_break,
        // field('operator', 'OR'),
        field('right', $.expression),
      )),
    ),

    binary_operator: $ => {
      const table = [
        [prec.left, '+', PREC.plus],
        [prec.left, '-', PREC.plus],
        [prec.left, '*', PREC.times],
        [prec.left, '@', PREC.times],
        [prec.left, '/', PREC.times],
        [prec.left, '%', PREC.times],
        [prec.left, '//', PREC.times],
        [prec.right, '**', PREC.power],
        [prec.left, '|', PREC.bitwise_or],
        [prec.left, '&', PREC.bitwise_and],
        [prec.left, '^', PREC.xor],
        [prec.left, '<<', PREC.shift],
        [prec.left, '>>', PREC.shift],
      ];
      
      // @ts-ignore
      return choice(...table.map(([fn, operator, precedence]) => fn(precedence, seq(
        field('left', $.primary_expression),
        // @ts-ignore
        field('operator', operator),
        field('right', $.primary_expression),
      ))));
    },

    unary_operator: $ => prec(PREC.unary, seq(
      field('operator', choice('+', '-', '~')),
      field('argument', $.primary_expression),
    )),

    subscript: $ => prec(PREC.call, seq(
      field('value', $.primary_expression),
      '[',
      commaSep1(field('subscript', $.expression)),
      optional(','),
      ']',
    )),

    parenthesized_expression: $ => prec(PREC.parenthesized_expression, seq(
      '(',
      $.expression,
      ')',
    )),

    basic_call: $ => seq(
      field('function', $.builtin_function),
      '(',
      field('parameters', sep1Break($.parameter, ',', $.parameter_break)),
      ')',
    ),

    builtin_function: $ => prec(20,choice(
      'datetimestamp', 
      'timetimestamp', 
      'concat', 
      'trim', 
    )),
    
    parameter: $ => choice(
      $.string,
      $.integer,
      $.float,
      $.true,
      $.false,
      $.spaces,
      $.timestamp,
      $.userid,
      $.attribute,
      $.group_subscript,
      $.group_last,
      $.unary_operator,
      $.subscript,
      $.basic_call,
    ),

    query_condition: $ => choice(
      $.query_comparison,
      $.query_boolean,
      $.parenthesized_condition,
    ),

    query_comparison: $ => prec.left(PREC.compare, seq(
      $.desired_attribute,
      repeat1(seq(
        field('operators',
          choice(
            '<',
            '<=',
            '=',
            '^=',
            '>=',
            '>',
            '<>',
            alias(seq('IS', 'LIKE'), 'IS_LIKE'),
          )),
        $.query_right_token,
      )),
    )),

    query_boolean: $ => choice(
      prec.left(PREC.and, seq(
        field('left', $.query_condition),
        $.booland_break,
        field('right', $.query_condition),
      )),
      prec.left(PREC.or, seq(
        field('left', $.query_condition),
        $.boolor_break,
        field('right', $.query_condition),
      )),
    ),

    desired_attribute: $ => seq(
      optional('DESIRED'),
      $.entity_attribute,
    ),
    
    current_attribute: $ => seq(
      'CURRENT',
      $.entity_attribute,
    ),

    query_right_token: $ => choice(
      $.string,
      $.integer,
      $.float,
      $.attribute,
      $.current_attribute,
    ),

    parenthesized_condition: $ => prec(PREC.parenthesized_expression, seq(
      '(',
      $.query_condition,
      ')',
    )),

    query_basic_call: $ => seq(
      field('function', $.aggregate_function),
      '(',
      field('parameter', $.identifier),
      ')',
    ),

    aggregate_function: $ => choice(
      'COUNT', 
      'MAX', 
      'MIN', 
      'AVERAGE',
      'SUM',
    ),

    string: $ => seq(
      $.string_start,
      $.string_content,
      $.string_end,
    ),
    
    string_start: _ => choice(/"/,/'/),
    string_content: _ => prec(PREC.string_content,/[^"']*/),
    string_end: _ => choice(/"/,/'/),

    string_text: $ => prec.right(repeat1(
      choice(
        $.escape_sequence,
        $._not_escape_sequence,
        $.string_content,
      )
    )),

    escape_sequence: _ => token.immediate(prec(1, seq(
      '\\',
      choice(
        /u[a-fA-F\d]{4}/,
        /U[a-fA-F\d]{8}/,
        /x[a-fA-F\d]{2}/,
        /\d{3}/,
        /\r?\n/,
        /['"abfrntv\\]/,
        /N\{[^}]+\}/,
      ),
    ))),

    _not_escape_sequence: _ => token.immediate('\\'),

    integer: _ => token(choice(
      seq(
	optional('-'),
        choice('0x', '0X'),
        repeat1(/_?[A-Fa-f0-9]+/),
        optional(/[Ll]/),
      ),
      seq(
	optional('-'),
        choice('0o', '0O'),
        repeat1(/_?[0-7]+/),
        optional(/[Ll]/),
      ),
      seq(
	optional('-'),
        choice('0b', '0B'),
        repeat1(/_?[0-1]+/),
        optional(/[Ll]/),
      ),
      seq(
        repeat1(/[0-9]+_?/),
        choice(
          optional(/[Ll]/), // long numbers
          optional(/[jJ]/), // complex numbers
        ),
      ),
    )),

    float: _ => {
      const digits = repeat1(/[0-9]+_?/);
      const exponent = seq(/[eE][\+-]?/, digits);

      return token(seq(
	optional('-'),
        choice(
          seq(digits, '.', optional(digits), optional(exponent)),
          seq(optional(digits), '.', digits, optional(exponent)),
          seq(digits, exponent),
        ),
        optional(choice(/[Ll]/, /[jJ]/)),
      ));
    },

    true: _ => 'True',
    false: _ => 'False',
    none: _ => 'None',
    spaces: _ => 'SPACES',
    timestamp: _ => 'CURRENT_TIMESTAMP',
    userid: _ => 'CLIENT_USER_ID',
    cblock: _ => '+->',
    lblock: _ => '+=>',
    csblock: _ => '+>',

    noteline: _ => /.+/,
    
    newline: _ => seq(optional('\r'), '\n'),

    line_continuation: _ => token(seq('\\', choice(seq(optional('\r'), '\n'), '\0'))),

  },   // end rules


});

module.exports.PREC = PREC;

/**
 * Creates a rule to match one or more of the rules separated by a comma
 *
 * @param {RuleOrLiteral} rule
 *
 * @return {SeqRule}
 *
 */
function commaSep1(rule) {
  return sep1(rule, ',');
}

/**
 * Creates a rule to match one or more occurrences of `rule` separated by `sep`
 *
 * @param {RuleOrLiteral} rule
 *
 * @param {RuleOrLiteral} separator
 *
 * @return {SeqRule}
 *
 */
function sep1(rule, separator) {
  return seq(rule, repeat(seq(separator, rule)));
}

function sep1Break(rule, separator, breakRule) {
  return seq(rule, repeat(seq(separator, optional(breakRule), rule)));
}
