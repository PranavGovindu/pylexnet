/* pylexnet (hand lane): run the hand-designed Python highlighter network.
 *
 * Two paths over the same weights file:
 *   default   the generated kernel in pylexnet_kernel.h (architecture only; values gathered from the file).
 *             The loader checks the structure hash and proves every guard inequality from the loaded values.
 *   --interp  the generic layered sparse evaluation (clip(W v + b, 0, hi) + carried entries), no kernel.
 * usage: pylexnet [-w weights.bin] [--interp] [--classes | --bench] [file]
 *   --classes writes one class byte (0..8) per input byte; without a file it reads stdin.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int64_t rows, cols, nskip, tied;     /* tied: one weight row and one ceiling shared by all rows */
    const double *W, *b, *hi;
    const int64_t *skip;
    int *rowptr, *col;
    double *val;
} Layer;

typedef struct {
    int64_t n_feats, n_fstate, n_fout, n_bstate, n_classes, head_cols, maxw;
    const double *emb, *h0, *head_W, *head_b;
    const int64_t *f_state, *f_out, *b_state;
    Layer *fwd, *bwd;
    int64_t n_fwd, n_bwd;
} Net;

static int use_interp = 0;

/* ---- loading ------------------------------------------------------------- */
typedef struct { const unsigned char *p, *end; } Cursor;

static const void *take(Cursor *c, size_t bytes) {
    if (c->p + bytes > c->end) { fprintf(stderr, "weights: truncated\n"); exit(1); }
    const void *r = c->p;
    c->p += bytes;
    return r;
}
static int64_t take_i64(Cursor *c) { int64_t v; memcpy(&v, take(c, 8), 8); return v; }

static Layer *load_cell(Cursor *c, int64_t *n, int64_t *maxw) {
    *n = take_i64(c);
    Layer *L = calloc((size_t)*n, sizeof(Layer));
    for (int64_t l = 0; l < *n; l++) {
        Layer *y = &L[l];
        y->rows = take_i64(c); y->cols = take_i64(c); y->nskip = take_i64(c); y->tied = take_i64(c);
        int64_t wrows = y->tied ? 1 : y->rows;
        y->W = take(c, (size_t)(wrows * y->cols) * 8);
        y->b = take(c, (size_t)y->rows * 8);
        y->hi = take(c, (size_t)wrows * 8);
        y->skip = take(c, (size_t)y->nskip * 8);
        if (y->cols > *maxw) *maxw = y->cols;
        if (y->rows + y->nskip > *maxw) *maxw = y->rows + y->nskip;
        int nnz = 0;
        for (int64_t i = 0; i < wrows * y->cols; i++) nnz += y->W[i] != 0;
        if (y->tied) nnz *= (int)y->rows;
        y->rowptr = malloc((size_t)(y->rows + 1) * sizeof(int));
        y->col = malloc((size_t)(nnz + 1) * sizeof(int));
        y->val = malloc((size_t)(nnz + 1) * sizeof(double));
        int k = 0;
        for (int64_t r = 0; r < y->rows; r++) {
            y->rowptr[r] = k;
            const double *w = y->W + (y->tied ? 0 : r) * y->cols;
            for (int64_t cc = 0; cc < y->cols; cc++) if (w[cc] != 0) y->col[k] = (int)cc, y->val[k++] = w[cc];
        }
        y->rowptr[y->rows] = k;
    }
    return L;
}

static Net load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *blob = malloc(n);
    if (fread(blob, 1, n, f) != n) { perror(path); exit(1); }
    fclose(f);
    Cursor c = { blob, blob + n };
    if (memcmp(take(&c, 8), "PYLX02\0\0", 8)) { fprintf(stderr, "%s: not a PYLX02 weight file\n", path); exit(1); }
    Net net = {0};
    net.n_feats = take_i64(&c); net.n_fstate = take_i64(&c); net.n_fout = take_i64(&c);
    net.n_bstate = take_i64(&c); net.n_classes = take_i64(&c); net.head_cols = take_i64(&c);
    net.emb = take(&c, (size_t)(128 * net.n_feats) * 8);
    net.maxw = net.n_feats + net.n_fstate;
    if (net.n_fout + net.n_bstate > net.maxw) net.maxw = net.n_fout + net.n_bstate;
    net.fwd = load_cell(&c, &net.n_fwd, &net.maxw);
    net.bwd = load_cell(&c, &net.n_bwd, &net.maxw);
    net.f_state = take(&c, (size_t)net.n_fstate * 8);
    net.f_out = take(&c, (size_t)net.n_fout * 8);
    net.b_state = take(&c, (size_t)net.n_bstate * 8);
    net.h0 = take(&c, (size_t)net.n_fstate * 8);
    net.head_W = take(&c, (size_t)net.head_cols * 8);      /* one integer-valued readout unit: the class id */
    net.head_b = take(&c, 8);
    if (c.p != c.end) { fprintf(stderr, "weights: %zu trailing bytes\n", (size_t)(c.end - c.p)); exit(1); }
    return net;
}

/* ---- generic inference --------------------------------------------------- */
static void run_cell(const Layer *layers, int64_t nl, double *v, double *tmp) {
    for (int64_t l = 0; l < nl; l++) {
        const Layer *L = &layers[l];
        for (int64_t r = 0; r < L->rows; r++) {
            double s = L->b[r];
            for (int k = L->rowptr[r]; k < L->rowptr[r + 1]; k++) s += L->val[k] * v[L->col[k]];
            double hi = L->hi[L->tied ? 0 : r];
            tmp[r] = s < 0 ? 0 : (s > hi ? hi : s);
        }
        for (int64_t k = 0; k < L->nskip; k++) tmp[L->rows + k] = v[L->skip[k]];
        memcpy(v, tmp, (size_t)(L->rows + L->nskip) * sizeof(double));
    }
}

/* n input bytes followed by one virtual newline, so a word at the end of the file still ends. */
static void classify_interp(const Net *net, const unsigned char *src, size_t n, unsigned char *cls, uint16_t *bits) {
    double *v = malloc((size_t)net->maxw * 8), *tmp = malloc((size_t)net->maxw * 8);
    double *h = malloc((size_t)net->n_fstate * 8), *r = calloc((size_t)net->n_bstate, 8);
    memcpy(h, net->h0, (size_t)net->n_fstate * 8);
    for (size_t t = 0; t <= n; t++) {
        unsigned c = t < n ? (src[t] < 128 ? src[t] : 127) : 10;
        memcpy(v, net->emb + c * net->n_feats, (size_t)net->n_feats * 8);
        memcpy(v + net->n_feats, h, (size_t)net->n_fstate * 8);
        run_cell(net->fwd, net->n_fwd, v, tmp);
        for (int64_t i = 0; i < net->n_fstate; i++) h[i] = v[net->f_state[i]];
        uint16_t bt = 0;
        for (int64_t i = 0; i < net->n_fout; i++) bt |= (uint16_t)((v[net->f_out[i]] != 0) << i);
        bits[t] = bt;
    }
    for (size_t t = n + 1; t-- > 0;) {
        for (int64_t i = 0; i < net->n_fout; i++) v[i] = (bits[t] >> i) & 1;
        memcpy(v + net->n_fout, r, (size_t)net->n_bstate * 8);
        run_cell(net->bwd, net->n_bwd, v, tmp);
        for (int64_t i = 0; i < net->n_bstate; i++) r[i] = v[net->b_state[i]];
        double s = net->head_b[0];
        for (int64_t c = 0; c < net->head_cols; c++) s += net->head_W[c] * v[c];
        s = s < 0 ? 0 : (s > net->n_classes - 1 ? net->n_classes - 1 : s);
        if (t < n) cls[t] = (unsigned char)s;
    }
    free(v); free(tmp); free(h); free(r);
}

/* ---- compiled mode ------------------------------------------------------- */
#include <emmintrin.h>
static inline double clip_sse(double x, double hi) { return _mm_cvtsd_f64(_mm_min_sd(_mm_max_sd(_mm_set_sd(x), _mm_setzero_pd()), _mm_set_sd(hi))); }
static inline double clip_fm(double x, double hi) { return __builtin_fmin(__builtin_fmax(x, 0.0), hi); }
#include "pylexnet_kernel.h"

typedef struct { uint64_t h; } Fnv;
static void fnv_bytes(Fnv *f, const void *p, size_t n) {
    for (size_t i = 0; i < n; i++) { f->h ^= ((const unsigned char *)p)[i]; f->h *= 0x100000001B3ULL; }
}
static void fnv_i64(Fnv *f, int64_t x) { fnv_bytes(f, &x, 8); }
static void fnv_mask(Fnv *f, const double *a, int64_t n) {
    for (int64_t i = 0; i < n; i++) { unsigned char m = a[i] != 0; fnv_bytes(f, &m, 1); }
}
static uint64_t struct_hash(const Net *net) {
    Fnv f = { 0xCBF29CE484222325ULL };
    fnv_i64(&f, net->n_feats); fnv_i64(&f, net->n_fstate); fnv_i64(&f, net->n_fout);
    fnv_i64(&f, net->n_bstate); fnv_i64(&f, net->n_classes); fnv_i64(&f, net->head_cols);
    const Layer *cells[2] = { net->fwd, net->bwd };
    int64_t counts[2] = { net->n_fwd, net->n_bwd };
    for (int c = 0; c < 2; c++) {
        fnv_i64(&f, counts[c]);
        for (int64_t l = 0; l < counts[c]; l++) {
            const Layer *L = &cells[c][l];
            fnv_i64(&f, L->rows); fnv_i64(&f, L->cols); fnv_i64(&f, L->nskip); fnv_i64(&f, L->tied);
            fnv_bytes(&f, L->hi, (size_t)(L->tied ? 1 : L->rows) * 8);
            for (int64_t k = 0; k < L->nskip; k++) fnv_i64(&f, L->skip[k]);
            fnv_mask(&f, L->W, (L->tied ? 1 : L->rows) * L->cols);
        }
    }
    for (int64_t i = 0; i < net->n_fstate; i++) fnv_i64(&f, net->f_state[i]);
    for (int64_t i = 0; i < net->n_fout; i++) fnv_i64(&f, net->f_out[i]);
    for (int64_t i = 0; i < net->n_bstate; i++) fnv_i64(&f, net->b_state[i]);
    fnv_mask(&f, net->head_W, net->head_cols);
    return f.h;
}

/* Every number in the file must be an integer: then all activations are exact integers in double arithmetic,
 * sums may be re-associated, and a clip1 unit is exactly 0 or 1. */
static int all_integers(const Net *net) {
    int bad = 0;
    for (int64_t i = 0; i < 128 * net->n_feats; i++) bad += net->emb[i] != (int64_t)net->emb[i];
    const Layer *cells[2] = { net->fwd, net->bwd };
    int64_t counts[2] = { net->n_fwd, net->n_bwd };
    for (int c = 0; c < 2; c++)
        for (int64_t l = 0; l < counts[c]; l++) {
            const Layer *L = &cells[c][l];
            int64_t wrows = L->tied ? 1 : L->rows;
            for (int64_t i = 0; i < wrows * L->cols; i++) bad += L->W[i] != (int64_t)L->W[i];
            for (int64_t i = 0; i < L->rows; i++) bad += L->b[i] != (int64_t)L->b[i];
            for (int64_t i = 0; i < wrows; i++) bad += L->hi[i] != (int64_t)L->hi[i];
        }
    for (int64_t i = 0; i < net->head_cols; i++) bad += net->head_W[i] != (int64_t)net->head_W[i];
    for (int64_t i = 0; i < net->n_fstate; i++) bad += net->h0[i] != (int64_t)net->h0[i];
    return bad;
}

/* Sorted threshold families: tied, one input, ceiling 1, biases monotone so the outputs form a step vector.
 * Then rank(h) by binary search reproduces every unit, and a consumer's weights over the family reduce to a
 * prefix sum indexed by the rank. Both tables are built here from the loaded values. */
static int check_families(const Net *net) {
    int bad = 0;
    for (int g = 0; g < K_N_FAMILY; g++) {
        int cell = K_FAMILY[g][0], l = K_FAMILY[g][1], n = K_FAMILY[g][3];
        const Layer *L = &(cell ? net->bwd : net->fwd)[l];
        int nz = 0;
        for (int64_t c = 0; c < L->cols; c++) nz += L->W[c] != 0;
        if (!L->tied || L->rows != n || L->hi[0] != 1.0 || nz != 1) { bad++; continue; }
        double w = 0;
        for (int64_t c = 0; c < L->cols; c++) if (L->W[c] != 0) w = L->W[c];
        for (int r = 1; r < n; r++) if ((w > 0 && L->b[r] > L->b[r - 1]) || (w < 0 && L->b[r] < L->b[r - 1])) bad++;
    }
    return bad;
}

static double *build_prefix(const Net *net) {
    double *kp = malloc(sizeof(double) * (K_PREFIX_LEN + 1));
    int pos = 0;
    for (int g = 0; g < K_N_PREFIX; g++) {
        int cell = K_PREFIX[g][0], l = K_PREFIX[g][1], r = K_PREFIX[g][2], c0 = K_PREFIX[g][3], n = K_PREFIX[g][4];
        const Layer *L = &(cell ? net->bwd : net->fwd)[l];
        kp[pos] = 0;
        for (int k = 0; k < n; k++) kp[pos + k + 1] = kp[pos + k] + L->W[(L->tied ? 0 : r) * L->cols + c0 + k];
        pos += n + 1;
    }
    return kp;
}

static void gather(const Net *net, double *kc, double *kb) {
    for (int i = 0; i < K_N_COEF; i++) {
        int cell = K_COEF[i][0], l = K_COEF[i][1], r = K_COEF[i][2], c = K_COEF[i][3];
        if (cell == 2) kc[i] = net->head_W[c];
        else { const Layer *L = &(cell ? net->bwd : net->fwd)[l]; kc[i] = L->W[(L->tied ? 0 : r) * L->cols + c]; }
    }
    for (int i = 0; i < K_N_BIAS; i++) {
        int cell = K_BIAS[i][0], l = K_BIAS[i][1], r = K_BIAS[i][2];
        kb[i] = cell == 2 ? net->head_b[r] : (cell ? net->bwd : net->fwd)[l].b[r];
    }
}

/* Upper bound of an entry of a layer's input vector, from the loaded ceilings and embedding table.
 * layer -1 is the cell's own input: byte features then carried state (forward), bits then state (backward). */
static double entry_max(const Net *net, int cell, int64_t layer, int64_t idx, int depth) {
    if (depth > 64) return 1e300;
    const Layer *L = cell ? net->bwd : net->fwd;
    if (layer < 0) {
        if (cell == 0) {
            if (idx < net->n_feats) {
                double m = 0;
                for (int b = 0; b < 128; b++) if (net->emb[b * net->n_feats + idx] > m) m = net->emb[b * net->n_feats + idx];
                return m;
            }
            return entry_max(net, 0, net->n_fwd - 1, net->f_state[idx - net->n_feats], depth + 1);
        }
        if (idx < net->n_fout) return 1.0;
        return entry_max(net, 1, net->n_bwd - 1, net->b_state[idx - net->n_fout], depth + 1);
    }
    if (idx < L[layer].rows) return L[layer].hi[L[layer].tied ? 0 : idx];
    return entry_max(net, cell, layer - 1, L[layer].skip[idx - L[layer].rows], depth + 1);
}

/* Each guarded unit must be provably 0 when its guard columns are 0: bias plus every other positive
 * contribution at its input's ceiling must not exceed 0. Returns the number of failed guards. */
static int check_guards(const Net *net) {
    int bad = 0;
    for (int g = 0; g < K_N_GUARD; g++) {
        int cell = K_GUARD[g][0], l = K_GUARD[g][1], r = K_GUARD[g][2];
        const Layer *L = &(cell ? net->bwd : net->fwd)[l];
        double bound = L->b[r];
        for (int64_t c = 0; c < L->cols; c++) {
            double w = L->W[(L->tied ? 0 : r) * L->cols + c];
            if (w <= 0) continue;
            int guarded = 0;
            for (int k = 0; k < K_MAX_GUARD; k++) if (K_GUARD[g][3 + k] == c) guarded = 1;
            if (!guarded) bound += w * entry_max(net, cell, l - 1, c, 0);
        }
        if (bound > 0) { bad++; fprintf(stderr, "guard %d (cell %d layer %d row %d) not provable: bound %g\n", g, cell, l, r, bound); }
    }
    return bad;
}

static void classify_compiled(const Net *net, const double *kc, const double *kb, const double *kp,
                              const unsigned char *src, size_t n, unsigned char *cls, uint16_t *bits) {
    FState a, b, *p = &a, *s = &b, *t;
    k_init_state(p, net->h0);
    for (size_t i = 0; i < n; i++) {
        bits[i] = k_fwd_step(kc, kb, kp, net->emb + (src[i] < 128 ? src[i] : 127) * K_N_FEATS, p, s);
        t = p; p = s; s = t;
    }
    bits[n] = k_fwd_step(kc, kb, kp, net->emb + 10 * K_N_FEATS, p, s);
    BState ra, rb, *r = &ra, *q = &rb, *u;
    memset(&ra, 0, sizeof ra);
    k_bwd_step(kc, kb, bits[n], r, q);
    u = r; r = q; q = u;
    for (size_t i = n; i-- > 0;) {
        cls[i] = (unsigned char)k_bwd_step(kc, kb, bits[i], r, q);
        u = r; r = q; q = u;
    }
}

/* ---- cli ----------------------------------------------------------------- */
static char *default_weights(const char *argv0) {
    const char *slash = strrchr(argv0, '/');
    size_t dir = slash ? (size_t)(slash - argv0 + 1) : 0;
    char *p = malloc(dir + 32);
    memcpy(p, argv0, dir);
    strcpy(p + dir, "pylexnet_weights.bin");
    return p;
}

int main(int argc, char **argv) {
    int classes = 0, bench = 0;
    const char *path = NULL, *wpath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--classes")) classes = 1;
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--interp")) use_interp = 1;
        else if (!strcmp(argv[i], "-w")) wpath = argv[++i];
        else path = argv[i];
    }
    Net net = load(wpath ? wpath : default_weights(argv[0]));
    double *kc = malloc(sizeof(double) * (K_N_COEF + 1)), *kb = malloc(sizeof(double) * (K_N_BIAS + 1)), *kp = NULL;
    if (!use_interp) {
        uint64_t h = struct_hash(&net);
        if (h != K_STRUCT_HASH) {
            fprintf(stderr, "weights: architecture hash %016llx does not match compiled kernel %016llx; regenerate (compile.py) or run --interp\n",
                    (unsigned long long)h, (unsigned long long)K_STRUCT_HASH);
            return 1;
        }
        if (check_guards(&net)) { fprintf(stderr, "weights: guard check failed, refusing the kernel\n"); return 1; }
        if (all_integers(&net)) { fprintf(stderr, "weights: non-integer values, refusing the kernel\n"); return 1; }
        if (check_families(&net)) { fprintf(stderr, "weights: a threshold family is not tied/sorted, refusing the kernel\n"); return 1; }
        gather(&net, kc, kb);
        kp = build_prefix(&net);
    }
    FILE *f = path ? fopen(path, "rb") : stdin;
    if (!f) { perror(path); return 1; }
    size_t cap = 1 << 20, n = 0, got;
    unsigned char *src = malloc(cap);
    while ((got = fread(src + n, 1, cap - n, f)) > 0) { n += got; if (n == cap) src = realloc(src, cap *= 2); }
    if (path) fclose(f);
    unsigned char *cls = malloc(n + 1);
    uint16_t *bits = malloc((n + 2) * sizeof(uint16_t));
    if (bench) {
        int reps = 5;
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int i = 0; i < reps; i++) {
            if (use_interp) classify_interp(&net, src, n, cls, bits);
            else classify_compiled(&net, kc, kb, kp, src, n, cls, bits);
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double sec = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
        fprintf(stderr, "%zu bytes, %.1f ns/byte\n", n, sec / reps / (double)n * 1e9);
        return 0;
    }
    if (use_interp) classify_interp(&net, src, n, cls, bits);
    else classify_compiled(&net, kc, kb, kp, src, n, cls, bits);
    if (classes) { fwrite(cls, 1, n, stdout); return 0; }
    static const char *ANSI[9] = {"", "\x1b[35m", "\x1b[36m", "\x1b[32m", "\x1b[2;3;37m", "\x1b[33m", "\x1b[34m", "\x1b[1;33m", "\x1b[38;5;208m"};
    int cur = 0;
    for (size_t t = 0; t < n; t++) {
        if (cls[t] != cur) { fputs("\x1b[0m", stdout); fputs(ANSI[cls[t]], stdout); cur = cls[t]; }
        putchar(src[t]);
    }
    fputs("\x1b[0m", stdout);
    return 0;
}
