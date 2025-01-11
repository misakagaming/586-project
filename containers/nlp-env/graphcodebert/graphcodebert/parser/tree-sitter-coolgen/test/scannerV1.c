#include "tree_sitter/parser.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define VEC_RESIZE(vec, _cap)                                                  \
    void *tmp = realloc((vec).data, (_cap) * sizeof((vec).data[0]));           \
    assert(tmp != NULL);                                                       \
    (vec).data = tmp;                                                          \
    (vec).cap = (_cap);

#define VEC_GROW(vec, _cap)                                                    \
    if ((vec).cap < (_cap)) {                                                  \
        VEC_RESIZE((vec), (_cap));                                             \
    }

#define VEC_PUSH(vec, el)                                                      \
    if ((vec).cap == (vec).len) {                                              \
        VEC_RESIZE((vec), MAX(16, (vec).len * 2));                             \
    }                                                                          \
    (vec).data[(vec).len++] = (el);

#define VEC_POP(vec) (vec).len--;

#define VEC_NEW                                                                \
    { .len = 0, .cap = 0, .data = NULL }

#define VEC_BACK(vec) ((vec).data[(vec).len - 1])

#define VEC_FREE(vec)                                                          \
    {                                                                          \
        if ((vec).data != NULL)                                                \
            free((vec).data);                                                  \
    }

#define VEC_CLEAR(vec) (vec).len = 0;

enum TokenType {
    NOTE_TERMINATOR,
    BLOCK_TERMINATOR,
    BOOLAND_BREAK,
    BOOLOR_BREAK,
    ERROR_SENTINEL,
};

typedef enum {
    SingleQuote = 1 << 0,
    DoubleQuote = 1 << 1,
    BackQuote = 1 << 2,
    Raw = 1 << 3,
    Format = 1 << 4,
    Triple = 1 << 5,
    Bytes = 1 << 6,
} Flags;

typedef struct {
    char flags;
} Delimiter;

static char ENTITY[] = "Entity";
static char GROUP[] = "Group";
static char WORK[] = "Work";

static inline Delimiter new_delimiter() { return (Delimiter){0}; }

static inline bool is_format(Delimiter *delimiter) {
    return delimiter->flags & Format;
}

static inline bool is_raw(Delimiter *delimiter) {
    return delimiter->flags & Raw;
}

static inline bool is_triple(Delimiter *delimiter) {
    return delimiter->flags & Triple;
}

static inline bool is_bytes(Delimiter *delimiter) {
    return delimiter->flags & Bytes;
}

static inline int32_t end_character(Delimiter *delimiter) {
    if (delimiter->flags & SingleQuote) {
        return '\'';
    }
    if (delimiter->flags & DoubleQuote) {
        return '"';
    }
    if (delimiter->flags & BackQuote) {
        return '`';
    }
    return 0;
}

static inline void set_format(Delimiter *delimiter) {
    delimiter->flags |= Format;
}

static inline void set_raw(Delimiter *delimiter) { delimiter->flags |= Raw; }

static inline void set_triple(Delimiter *delimiter) {
    delimiter->flags |= Triple;
}

static inline void set_bytes(Delimiter *delimiter) {
    delimiter->flags |= Bytes;
}

static inline void set_end_character(Delimiter *delimiter, int32_t character) {
    switch (character) {
        case '\'':
            delimiter->flags |= SingleQuote;
            break;
        case '"':
            delimiter->flags |= DoubleQuote;
            break;
        case '`':
            delimiter->flags |= BackQuote;
            break;
        default:
            assert(false);
    }
}

static inline bool isDigit(int32_t character) {
    return character >= '0' && character <= '9';
}

typedef struct {
    uint32_t len;
    uint32_t cap;
    uint16_t *data;
} indent_vec;

static indent_vec indent_vec_new() {
    indent_vec vec = VEC_NEW;
    vec.data = calloc(1, sizeof(uint16_t));
    vec.cap = 1;
    return vec;
}

typedef struct {
    uint32_t len;
    uint32_t cap;
    Delimiter *data;
} delimiter_vec;

static delimiter_vec delimiter_vec_new() {
    delimiter_vec vec = VEC_NEW;
    vec.data = calloc(1, sizeof(Delimiter));
    vec.cap = 1;
    return vec;
}

typedef struct {
    indent_vec indents;
    delimiter_vec delimiters;
    bool inside_f_string;
} Scanner;

static inline void advance(TSLexer *lexer) { lexer->advance(lexer, false); }

static inline void skip(TSLexer *lexer) { lexer->advance(lexer, true); }

bool tree_sitter_coolgen_external_scanner_scan(void *payload, TSLexer *lexer,
                                              const bool *valid_symbols) {
    Scanner *scanner = (Scanner *)payload;

    bool error_recovery_mode = valid_symbols[ERROR_SENTINEL];

    lexer->mark_end(lexer);

    if (valid_symbols[NOTE_TERMINATOR] && !error_recovery_mode) {
    
	while (lexer->lookahead && lexer->lookahead == ' ') {
           skip(lexer);
        }

	if (lexer->lookahead && isDigit(lexer->lookahead)) {
	  while (lexer->lookahead && isDigit(lexer->lookahead)) {
            skip(lexer);
          }

	  while (lexer->lookahead && 
                   (lexer->lookahead == ' ' || 
                    lexer->lookahead == '!' || 
                    lexer->lookahead == '.' || 
                    lexer->lookahead == '\r')) {
            skip(lexer);
          }
	  
	  if (lexer->lookahead && lexer->lookahead == '\n') {
            lexer->result_symbol = NOTE_TERMINATOR;
            skip(lexer);
	    lexer->mark_end(lexer);
            return true;
	  }
	}
    }

    if (valid_symbols[BLOCK_TERMINATOR] && !error_recovery_mode) {
    
	while (lexer->lookahead && lexer->lookahead == ' ') {
           skip(lexer);
        }
	
	if (lexer->lookahead && isDigit(lexer->lookahead)) {
	  while (lexer->lookahead && isDigit(lexer->lookahead)) {
            skip(lexer);
          }
	}

	while (lexer->lookahead && (lexer->lookahead == ' ' || lexer->lookahead == '!' || lexer->lookahead == '\r')) {
            skip(lexer);
        }
	  
	if (lexer->lookahead && (lexer->lookahead == '\n' || lexer->lookahead == '+')) {
	    lexer->mark_end(lexer);
            lexer->result_symbol = BLOCK_TERMINATOR;
	    if (lexer->lookahead == '\n') {
               skip(lexer);
	       lexer->mark_end(lexer);
            }
            return true;
	}
    }

    if ((valid_symbols[BOOLAND_BREAK] || valid_symbols[BOOLOR_BREAK]) && !error_recovery_mode) {
   
	while (lexer->lookahead && (lexer->lookahead == ' ' || lexer->lookahead == '\n' || lexer->lookahead == '\r')) {
             skip(lexer);
        }

        if (lexer->lookahead && isDigit(lexer->lookahead)) {
	  while (lexer->lookahead && isDigit(lexer->lookahead)) {
            skip(lexer);
          }
        }

	while (lexer->lookahead && (lexer->lookahead == ' ' || lexer->lookahead == '!' || lexer->lookahead == '\r')) {
            skip(lexer);
        }
	   
        if (lexer->lookahead && lexer->lookahead == 'A') {
          advance(lexer);
          if (lexer->lookahead && lexer->lookahead == 'N') {
             advance(lexer);
             if (lexer->lookahead && lexer->lookahead == 'D') {
                advance(lexer);
	        lexer->mark_end(lexer);
                lexer->result_symbol = BOOLAND_BREAK;
                return true;
             }
          }
        }
        
        if (lexer->lookahead && lexer->lookahead == 'O') {
           advance(lexer);
           if (lexer->lookahead && lexer->lookahead == 'R') {
              advance(lexer);
	      lexer->mark_end(lexer);
              lexer->result_symbol = BOOLOR_BREAK;
              return true;
           }
        }
    } 

    return false;
}

unsigned tree_sitter_coolgen_external_scanner_serialize(void *payload,
                                                       char *buffer) {
    Scanner *scanner = (Scanner *)payload;

    size_t size = 0;

    buffer[size++] = (char)scanner->inside_f_string;

    size_t delimiter_count = scanner->delimiters.len;
    if (delimiter_count > UINT8_MAX) {
        delimiter_count = UINT8_MAX;
    }
    buffer[size++] = (char)delimiter_count;

    if (delimiter_count > 0) {
        memcpy(&buffer[size], scanner->delimiters.data, delimiter_count);
    }
    size += delimiter_count;

    int iter = 1;
    for (; iter < scanner->indents.len &&
           size < TREE_SITTER_SERIALIZATION_BUFFER_SIZE;
         ++iter) {
        buffer[size++] = (char)scanner->indents.data[iter];
    }

    return size;
}

void tree_sitter_coolgen_external_scanner_deserialize(void *payload,
                                                     const char *buffer,
                                                     unsigned length) {
    Scanner *scanner = (Scanner *)payload;

    VEC_CLEAR(scanner->delimiters);
    VEC_CLEAR(scanner->indents);
    VEC_PUSH(scanner->indents, 0);

    if (length > 0) {
        size_t size = 0;

        scanner->inside_f_string = (bool)buffer[size++];

        size_t delimiter_count = (uint8_t)buffer[size++];
        if (delimiter_count > 0) {
            VEC_GROW(scanner->delimiters, delimiter_count);
            scanner->delimiters.len = delimiter_count;
            memcpy(scanner->delimiters.data, &buffer[size], delimiter_count);
            size += delimiter_count;
        }

        for (; size < length; size++) {
            VEC_PUSH(scanner->indents, (unsigned char)buffer[size]);
        }
    }
}

void *tree_sitter_coolgen_external_scanner_create() {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    _Static_assert(sizeof(Delimiter) == sizeof(char), "");
#else
    assert(sizeof(Delimiter) == sizeof(char));
#endif
    Scanner *scanner = calloc(1, sizeof(Scanner));
    scanner->indents = indent_vec_new();
    scanner->delimiters = delimiter_vec_new();
    tree_sitter_coolgen_external_scanner_deserialize(scanner, NULL, 0);
    return scanner;
}

void tree_sitter_coolgen_external_scanner_destroy(void *payload) {
    Scanner *scanner = (Scanner *)payload;
    VEC_FREE(scanner->indents);
    VEC_FREE(scanner->delimiters);
    free(scanner);
}
