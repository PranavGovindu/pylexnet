/* pylexnet: a recurrent network with hand-compiled weights that classifies Python bytes.
 * This runtime knows nothing about Python: it only does sparse matrix products and ReLU.
 *
 * The forward cell is a stack of sparse blocks with skip connections over one global
 * activation vector: inputs [state | byte one-hot], then each block's rows in order.
 * Block k computes y_k = act(W_k a + b_k) where a is everything computed so far.
 * The last block's first rows are the next state; then the label and event one-hots.
 *
 * Two kernels compute the same products:
 *   --reference  the plain one: scatter every active column, track touched rows with flags.
 *   default      splits the columns of every block into "known" and "unknown" ones.
 *                The byte one-hot columns are known: the runtime builds that input itself,
 *                so their values given the byte are known before any weight is read. A row
 *                whose every entry sits in known columns is a pure function of the byte, so
 *                its column is known too (B1's byte-class rows, for instance). At load time
 *                the kernel evaluates, for each of the 256 bytes, the known part of every
 *                row: K_i(b) = sum of w_ij * a_j(b) over known j, and act(b_i + K_i(b)).
 *                Per byte it then scatters only the unknown columns (the state and the rows
 *                derived from it), and completes each touched row with its K_i(b). Rows not
 *                touched by any unknown column produce act(b_i + K_i(b)), which is the
 *                precomputed value; the ones where it is non-zero are emitted from a per-byte
 *                "loud" list. Every row still receives exactly W_i . a + b_i; the split only
 *                moves the one-hot part of the product to load time.
 *                On top of that, dead-row pruning: the weights file declares a bound U on every
 *                activation. For byte b, a ReLU row whose bias + K_i(b) + (largest possible
 *                contribution of its unknown columns under U) is still <= 0 is provably zero, so
 *                its entries are left out of that byte's scatter, and its own column counts as
 *                zero for the rows downstream. The kernel checks every activation it produces
 *                against U; if one ever exceeds it, the file is redone with --reference. Nothing
 *                here depends on the state: the state still flows through every block's weights. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { int i; float w; } Ent;
typedef struct { int c; float v; } Act;
typedef struct {
    int rows, lo, hi, nnz, relu, base;   /* columns [lo, hi) of the global vector; rows live at base */
    const int *ptr, *idx; const float *val, *bias;
    int *touched, *brow, nb; char *seen; float *y;
    /* fast kernel */
    int *uptr, *uidx; float *uval;       /* CSC over the unknown columns only */
    int *kptr, *kcol; float *kval;       /* CSR, per row, of its entries in known columns */
    int *lptr, *lrow; float *lval;       /* per byte: rows with act(bias + K_i(b)) != 0, and that value */
    signed char *ktab; int hask;         /* K_i(b) as a dense int8 table [b * rows + i] when it fits, for rows touched by unknown columns */
    int nvar, *vmap;                     /* per byte: which pruned CSC variant to scatter with */
    int **vptr, **vidx; float **vval;    /* pruned CSC variants: only entries of rows that can be non-zero for that byte */
    Ent **vent;                          /* the same entries interleaved (row, weight); vptr has a sentinel empty column at index w */
} Block;
static const unsigned char *P;
static const void *take(size_t n) { const void *r = P; P += n; return r; }
static void load(Block *l) {
    const int *h = take(24);
    l->rows = h[0]; l->lo = h[1]; l->hi = h[2]; l->nnz = h[3]; l->relu = h[4]; l->base = h[5];
    l->ptr = take(4 * (size_t)(l->hi - l->lo + 1));
    l->idx = take(4 * (size_t)l->nnz);
    l->val = take(4 * (size_t)l->nnz);
    l->bias = take(4 * (size_t)l->rows);
    l->touched = malloc(sizeof(int) * (size_t)(l->rows + l->nnz));
    l->brow = malloc(sizeof(int) * (size_t)l->rows);
    l->seen = calloc((size_t)l->rows, 1);
    l->y = malloc(sizeof(float) * (size_t)l->rows);
    memcpy(l->y, l->bias, sizeof(float) * (size_t)l->rows);
    l->nb = 0;
    for (int i = 0; i < l->rows; i++) if (l->bias[i] > 0 || (!l->relu && l->bias[i] != 0)) l->brow[l->nb++] = i;
}
static inline float act(int relu, float v) { return relu && v < 0 ? 0 : v; }

/* Known-column analysis. known[j] for every global column; kv[b][...] sparse lists of the
 * known columns' values for byte b. */
static char *g_known;
static int *g_kvptr, *g_kvcol; static float *g_kvval; static int g_kvn, g_kvcap;
static int g_needkv;
static float g_U;        /* declared |activation| bound */
static float g_vmax;     /* largest |activation| the fast kernel has produced */
static char *g_nonneg;   /* column is a ReLU output */
static char *g_dead;     /* per byte, columns that are provably zero (dead rows of earlier blocks) */
static long long g_tb[16]; /* touched rows per block (bench) */
static float *g_kv;  /* dense scratch over the global vector: values of known columns for the current byte */
static void kv_push(int col, float v) {
    if (g_kvn == g_kvcap) { g_kvcap = g_kvcap ? 2 * g_kvcap : 4096; g_kvcol = realloc(g_kvcol, sizeof(int) * (size_t)g_kvcap); g_kvval = realloc(g_kvval, sizeof(float) * (size_t)g_kvcap); }
    g_kvcol[g_kvn] = col; g_kvval[g_kvn++] = v;
}
static void specialise(Block *L, int NB, int total, int oB) {
    g_known = calloc((size_t)total, 1);
    g_nonneg = calloc((size_t)total, 1);
    g_dead = calloc((size_t)total * 256, 1);   /* [b * total + col]: column provably zero for byte b */
    g_kv = calloc((size_t)total, sizeof(float));
    for (int b = 0; b < 256; b++) g_known[oB + b] = 1;
    /* per byte lists of known values; byte columns first */
    /* temporary per-byte dynamic lists for known values, built block by block */
    int *tcap = calloc(256, sizeof(int)), *tn = calloc(256, sizeof(int));
    int **tcol = calloc(256, sizeof(int *)); float **tval = calloc(256, sizeof(float *));
    for (int b = 0; b < 256; b++) { tcap[b] = 64; tcol[b] = malloc(sizeof(int) * 64); tval[b] = malloc(sizeof(float) * 64); tcol[b][0] = oB + b; tval[b][0] = 1; tn[b] = 1; }
    #define TPUSH(b, c, v) do { if (tn[b] == tcap[b]) { tcap[b] *= 2; tcol[b] = realloc(tcol[b], sizeof(int) * (size_t)tcap[b]); tval[b] = realloc(tval[b], sizeof(float) * (size_t)tcap[b]); } tcol[b][tn[b]] = (c); tval[b][tn[b]++] = (v); } while (0)
    for (int k = 0; k < NB; k++) {
        Block *l = &L[k];
        int w = l->hi - l->lo;
        /* row degrees in known columns, and whether the row has any unknown entry */
        int *kdeg = calloc((size_t)l->rows, sizeof(int));
        char *hasu = calloc((size_t)l->rows, 1);
        int nu = 0;
        for (int j = 0; j < w; j++)
            for (int e = l->ptr[j]; e < l->ptr[j + 1]; e++) {
                if (g_known[l->lo + j]) kdeg[l->idx[e]]++; else { hasu[l->idx[e]] = 1; nu++; }
            }
        /* CSC over unknown columns */
        l->uptr = malloc(sizeof(int) * (size_t)(w + 1)); l->uidx = malloc(sizeof(int) * (size_t)(nu + 1)); l->uval = malloc(sizeof(float) * (size_t)(nu + 1));
        /* CSR over known columns */
        l->kptr = malloc(sizeof(int) * (size_t)(l->rows + 1));
        l->kptr[0] = 0;
        for (int i = 0; i < l->rows; i++) l->kptr[i + 1] = l->kptr[i] + kdeg[i];
        l->kcol = malloc(sizeof(int) * (size_t)(l->kptr[l->rows] + 1)); l->kval = malloc(sizeof(float) * (size_t)(l->kptr[l->rows] + 1));
        int *fill = calloc((size_t)l->rows, sizeof(int));
        nu = 0;
        for (int j = 0; j < w; j++) {
            l->uptr[j] = nu;
            for (int e = l->ptr[j]; e < l->ptr[j + 1]; e++) {
                int i = l->idx[e];
                if (g_known[l->lo + j]) { l->kcol[l->kptr[i] + fill[i]] = l->lo + j; l->kval[l->kptr[i] + fill[i]++] = l->val[e]; }
                else { l->uidx[nu] = i; l->uval[nu++] = l->val[e]; }
            }
        }
        l->uptr[w] = nu;
        /* rows with no unknown entry are known columns themselves */
        for (int i = 0; i < l->rows; i++) if (!hasu[i]) g_known[l->base + i] = 1;
        /* per byte: K_i(b) for every row with known entries; loud rows; values of known rows */
        l->lptr = malloc(sizeof(int) * 257);
        int lcap = 1024, ln = 0;
        l->lrow = malloc(sizeof(int) * (size_t)lcap); l->lval = malloc(sizeof(float) * (size_t)lcap);
        for (int b = 0; b < 256; b++) {
            l->lptr[b] = ln;
            for (int a = 0; a < tn[b]; a++) g_kv[tcol[b][a]] = tval[b][a];
            for (int i = 0; i < l->rows; i++) {
                float K = 0;
                for (int e = l->kptr[i]; e < l->kptr[i + 1]; e++) K += l->kval[e] * g_kv[l->kcol[e]];
                float v = act(l->relu, l->bias[i] + K);
                if (!hasu[i] && v != 0) TPUSH(b, l->base + i, v);
                if (v != 0 && (hasu[i] || k == NB - 1)) {   /* known rows are consumed through K only; last block's are the outputs */
                    if (ln == lcap) { lcap *= 2; l->lrow = realloc(l->lrow, sizeof(int) * (size_t)lcap); l->lval = realloc(l->lval, sizeof(float) * (size_t)lcap); }
                    l->lrow[ln] = i; l->lval[ln++] = v;
                }
            }
            for (int a = 0; a < tn[b]; a++) g_kv[tcol[b][a]] = 0;
        }
        l->lptr[256] = ln;
        /* dense int8 table of K_i(b) for rows that mix known and unknown entries */
        l->hask = 0; l->ktab = NULL;
        for (int i = 0; i < l->rows; i++) if (hasu[i] && kdeg[i]) l->hask = 1;
        if (l->hask) {
            int ok = 1;
            signed char *tab = calloc((size_t)l->rows * 256, 1);
            for (int b = 0; b < 256 && ok; b++) {
                for (int a = 0; a < tn[b]; a++) g_kv[tcol[b][a]] = tval[b][a];
                for (int i = 0; i < l->rows; i++) {
                    if (!hasu[i]) continue;
                    float K = 0;
                    for (int e = l->kptr[i]; e < l->kptr[i + 1]; e++) K += l->kval[e] * g_kv[l->kcol[e]];
                    if (K != (float)(int)K || K < -127 || K > 127) ok = 0;
                    tab[(size_t)b * l->rows + i] = (signed char)(int)K;
                }
                for (int a = 0; a < tn[b]; a++) g_kv[tcol[b][a]] = 0;
            }
            if (ok) l->ktab = tab; else free(tab);
        }
        /* dead-row pruning: for byte b, row i can only be non-zero if
         *   bias_i + K_i(b) + sum over unknown entries of (max contribution under the bound) > 0   (ReLU rows)
         * Entries of rows that cannot be non-zero are left out of the scatter for that byte. The
         * pruned column structures are deduplicated across bytes. */
        {
            int *aptr = malloc(sizeof(int) * (size_t)(w + 1)), *aidx = malloc(sizeof(int) * (size_t)(nu + 1));
            float *aval = malloc(sizeof(float) * (size_t)(nu + 1));
            float *slack = malloc(sizeof(float) * (size_t)l->rows);   /* max unknown contribution per row, for this byte */
            char *alive = malloc((size_t)l->rows);
            l->nvar = 0; l->vmap = malloc(sizeof(int) * 256);
            l->vptr = malloc(sizeof(int *) * 256); l->vidx = malloc(sizeof(int *) * 256); l->vval = malloc(sizeof(float *) * 256);
            l->vent = malloc(sizeof(Ent *) * 256);
            for (int b = 0; b < 256; b++) {
                const char *db = g_dead + (size_t)b * total;
                for (int i = 0; i < l->rows; i++) slack[i] = 0;
                for (int j = 0; j < w; j++) {
                    if (db[l->lo + j]) continue;   /* a column that is provably zero for this byte contributes nothing */
                    for (int e = l->uptr[j]; e < l->uptr[j + 1]; e++) {
                        float wv = l->uval[e];
                        slack[l->uidx[e]] += (g_nonneg[l->lo + j] ? (wv > 0 ? wv : 0) : (wv < 0 ? -wv : wv)) * g_U;
                    }
                }
                for (int a = 0; a < tn[b]; a++) g_kv[tcol[b][a]] = tval[b][a];
                for (int i = 0; i < l->rows; i++) {
                    float K = 0;
                    for (int e = l->kptr[i]; e < l->kptr[i + 1]; e++) K += l->kval[e] * g_kv[l->kcol[e]];
                    alive[i] = l->relu ? (l->bias[i] + K + slack[i] > 0) : 1;
                    if (!alive[i] && hasu[i]) g_dead[(size_t)b * total + l->base + i] = 1;
                }
                for (int a = 0; a < tn[b]; a++) g_kv[tcol[b][a]] = 0;
                int na = 0;
                for (int j = 0; j < w; j++) {
                    aptr[j] = na;
                    if (db[l->lo + j]) continue;
                    for (int e = l->uptr[j]; e < l->uptr[j + 1]; e++)
                        if (alive[l->uidx[e]]) { aidx[na] = l->uidx[e]; aval[na++] = l->uval[e]; }
                }
                aptr[w] = na;
                int v;
                for (v = 0; v < l->nvar; v++)
                    if (l->vptr[v][w] == na && !memcmp(l->vptr[v], aptr, sizeof(int) * (size_t)(w + 1)) &&
                        !memcmp(l->vidx[v], aidx, sizeof(int) * (size_t)na) && !memcmp(l->vval[v], aval, sizeof(float) * (size_t)na)) break;
                if (v == l->nvar) {
                    l->vptr[v] = malloc(sizeof(int) * (size_t)(w + 2)); memcpy(l->vptr[v], aptr, sizeof(int) * (size_t)(w + 1));
                    l->vptr[v][w + 1] = na;   /* sentinel: column w is empty, used for columns outside [lo, hi) */
                    l->vent[v] = malloc(sizeof(Ent) * (size_t)(na + 1));
                    for (int e = 0; e < na; e++) { l->vent[v][e].i = aidx[e]; l->vent[v][e].w = aval[e]; }
                    l->vidx[v] = malloc(sizeof(int) * (size_t)(na + 1)); memcpy(l->vidx[v], aidx, sizeof(int) * (size_t)na);
                    l->vval[v] = malloc(sizeof(float) * (size_t)(na + 1)); memcpy(l->vval[v], aval, sizeof(float) * (size_t)na);
                    l->nvar++;
                }
                l->vmap[b] = v;
            }
            free(aptr); free(aidx); free(aval); free(slack); free(alive);
        }
        if (l->relu) for (int i = 0; i < l->rows; i++) g_nonneg[l->base + i] = 1;
        free(kdeg); free(hasu); free(fill);
    }
    g_kvptr = malloc(sizeof(int) * 257);
    for (int b = 0; b < 256; b++) { g_kvptr[b] = g_kvn; for (int a = 0; a < tn[b]; a++) kv_push(tcol[b][a], tval[b][a]); }
    g_kvptr[256] = g_kvn;
    g_needkv = 0;
    for (int k = 0; k < NB; k++) if (L[k].hask && !L[k].ktab) g_needkv = 1;
    for (int b = 0; b < 256; b++) { free(tcol[b]); free(tval[b]); }
    free(tcol); free(tval); free(tn); free(tcap);
}

static long long g_touched;
static Block *g_L;
static int g_ref;
/* y = act(W a + b), visiting only the non-zero entries of a. Appends the non-zero rows
 * (as global indices) to the active list. */
static int apply_ref(Block *l, const int *in, const float *inv, int nin, int *out, float *outv) {
    float *y = l->y;
    /* y holds the bias outside this call; touched rows are restored afterwards */
    int nt = 0;
    for (int a = 0; a < nin; a++) {
        int j = in[a] - l->lo;
        if (j < 0 || j >= l->hi - l->lo) continue;
        float v = inv[a];
        for (int e = l->ptr[j]; e < l->ptr[j + 1]; e++) {
            int i = l->idx[e];
            if (!l->seen[i]) { l->seen[i] = 1; l->touched[nt++] = i; }
            y[i] += l->val[e] * v;
        }
    }
    for (int a = 0; a < l->nb; a++) {
        int i = l->brow[a];
        if (!l->seen[i]) { l->seen[i] = 1; l->touched[nt++] = i; }
    }
    g_touched += nt;
    int n = 0;
    for (int a = 0; a < nt; a++) {
        int i = l->touched[a];
        float v = y[i];
        if (l->relu && v < 0) v = 0;
        if (v != 0) { out[n] = l->base + i; outv[n++] = v; }
        y[i] = l->bias[i]; l->seen[i] = 0;
    }
    return n;
}

/* Fast kernel. y rests at the bias; a row is touched iff y != bias (a row equal to its bias
 * yields exactly what an untouched row yields, so duplicates in the touched list and
 * cancellations are harmless). The byte's known values are in g_kv. */
static inline int apply(Block *restrict l, int b, const Act *restrict in, int nin, Act *restrict out) {
    float *restrict y = l->y;
    const float *restrict bias = l->bias;
    int var = l->vmap[b];
    const int *restrict ptr = l->vptr[var];
    const Ent *restrict ent = l->vent[var];
    int *restrict tl = l->touched;
    int nt = 0, lo = l->lo, w = l->hi - l->lo, base = l->base, relu = l->relu;
    float vmax = g_vmax;
    for (int a = 0; a < nin; a++) {
        unsigned j = (unsigned)(in[a].c - lo);
        j = j < (unsigned)w ? j : (unsigned)w;    /* outside the block's columns: the empty sentinel column */
        float v = in[a].v;
        int e = ptr[j], ee = ptr[j + 1];
        for (; e < ee; e++) { int i = ent[e].i; y[i] += ent[e].w * v; tl[nt++] = i; }
    }
    g_touched += nt; g_tb[l - g_L] += nt;
    int n = 0;
    /* rows that are non-zero from their known part alone, unless an unknown column touched them */
    for (int a = l->lptr[b], ae = l->lptr[b + 1]; a < ae; a++) {
        int i = l->lrow[a];
        float v = l->lval[a];
        out[n].c = base + i; out[n].v = v; n += y[i] == bias[i]; vmax = fabsf(v) > vmax ? fabsf(v) : vmax;
    }
    if (!l->hask) {
        if (relu) {
            for (int a = 0; a < nt; a++) {
                int i = tl[a];
                float v = y[i], bi = bias[i]; y[i] = bi;
                v = (v != bi) ? v : 0;
                out[n].c = base + i; out[n].v = v; n += v > 0; vmax = v > vmax ? v : vmax;
            }
        } else {
            for (int a = 0; a < nt; a++) {
                int i = tl[a];
                float v = y[i], bi = bias[i]; y[i] = bi;
                v = (v != bi) ? v : 0;
                out[n].c = base + i; out[n].v = v; n += v != 0; vmax = fabsf(v) > vmax ? fabsf(v) : vmax;
            }
        }
    } else if (l->ktab) {
        const signed char *restrict kt = l->ktab + (size_t)b * l->rows;
        if (relu) {
            for (int a = 0; a < nt; a++) {
                int i = tl[a];
                float v = y[i], bi = bias[i]; y[i] = bi;
                v = (v != bi) ? v + (float)kt[i] : 0;   /* at rest: contributes nothing here (loud list covers it) */
                out[n].c = base + i; out[n].v = v; n += v > 0; vmax = v > vmax ? v : vmax;
            }
        } else {
            for (int a = 0; a < nt; a++) {
                int i = tl[a];
                float v = y[i], bi = bias[i]; y[i] = bi;
                v = (v != bi) ? v + (float)kt[i] : 0;
                out[n].c = base + i; out[n].v = v; n += v != 0; vmax = fabsf(v) > vmax ? fabsf(v) : vmax;
            }
        }
    } else {
        const int *restrict kptr = l->kptr, *restrict kcol = l->kcol;
        const float *restrict kval = l->kval, *restrict kv = g_kv;
        for (int a = 0; a < nt; a++) {
            int i = tl[a];
            float v = y[i], bi = bias[i];
            if (v != bi) {
                y[i] = bi;
                for (int e = kptr[i], ee = kptr[i + 1]; e < ee; e++) v += kval[e] * kv[kcol[e]];
                if (relu && v < 0) v = 0;
                if (v != 0) { out[n].c = base + i; out[n].v = v; n++; if (fabsf(v) > vmax) vmax = fabsf(v); }
            }
        }
    }
    g_vmax = vmax;
    return n;
}
static const char *ANSI[9] = {"", "\x1b[1;35m", "\x1b[36m", "\x1b[32m", "\x1b[2;3;37m", "\x1b[33m", "\x1b[1;34m", "\x1b[1;33m", "\x1b[38;5;208m"};

int main(int argc, char **argv) {
    const char *wpath = "pylexnet_weights.bin", *path = NULL;
    int classes = 0, bench = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--classes")) classes = 1;
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--reference")) g_ref = 1;
        else if (!strcmp(argv[i], "-w")) wpath = argv[++i];
        else path = argv[i];
    }
    FILE *f = fopen(wpath, "rb");
    if (!f) { perror(wpath); return 1; }
    fseek(f, 0, SEEK_END); long ws = ftell(f); rewind(f);
    unsigned char *wb = malloc((size_t)ws);
    if (fread(wb, 1, (size_t)ws, f) != (size_t)ws) return 1;
    fclose(f);
    P = wb;
    const int *h = take(40);
    int NIN = h[0], NSTATE = h[1], NB = h[2], ninit = h[3], oL = h[4], LAB = h[5], oE = h[6], EV = h[7], oB = h[8];
    g_U = (float)h[9];   /* declared bound on every activation; verified at runtime, see check below */
    const int *init = take(4 * (size_t)ninit);
    const int *evmap = take(4 * (size_t)(EV - 9) * 9);  /* events >= 9 map the backward register instead of setting it */
    Block *L = malloc(sizeof(Block) * (size_t)NB);
    g_L = L;
    int total = NIN;
    for (int k = 0; k < NB; k++) { load(&L[k]); total += L[k].rows; }
    if (!g_ref) specialise(L, NB, total, oB);
    f = path ? fopen(path, "rb") : stdin;
    if (!f) { perror(path); return 1; }
    size_t cap = 1 << 20, n = 0;
    unsigned char *src = malloc(cap);
    size_t got;
    while ((got = fread(src + n, 1, cap - n, f)) > 0) { n += got; if (n == cap) src = realloc(src, cap *= 2); }

    float *av = malloc(sizeof(float) * (size_t)total);
    int *ai = malloc(sizeof(int) * (size_t)total);
    Act *A0 = malloc(sizeof(Act) * (size_t)total), *A1 = malloc(sizeof(Act) * (size_t)total);
    unsigned char *lab = malloc(n + 1); signed char *ev = malloc(n + 2);
    int *st = malloc(sizeof(int) * (size_t)NSTATE); float *sv = malloc(sizeof(float) * (size_t)NSTATE);
    Block *last = &L[NB - 1];
    int lbase = last->base, lim = oL + LAB;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int pass = 0; pass < 2; pass++) {
        int ref = g_ref || pass;
        g_vmax = 0;
        if (ref) {
            int ns = ninit;
            for (int i = 0; i < ninit; i++) { st[i] = init[i]; sv[i] = 1; }
            for (size_t t = 0; t <= n; t++) {
                unsigned char b = t < n ? src[t] : 10;
                int na = 0;
                for (int i = 0; i < ns; i++) { ai[na] = st[i]; av[na++] = sv[i]; }
                ai[na] = oB + b; av[na++] = 1;
                int from = 0;
                for (int k = 0; k < NB; k++) { from = na; na += apply_ref(&L[k], ai, av, na, ai + na, av + na); }
                ns = 0; int l = 0, e = -1;
                for (int i = from; i < na; i++) {
                    int r = ai[i] - lbase;
                    if (r < NSTATE) { st[ns] = r; sv[ns++] = av[i]; }
                    else if (r < lim) l = r - oL;
                    else e = r - oE;
                }
                if (t < n) lab[t] = (unsigned char)l;
                ev[t] = (signed char)e;
            }
        } else {
            Act *cur = A0, *nxt = A1;
            int ns = ninit;
            for (int i = 0; i < ninit; i++) { cur[i].c = init[i]; cur[i].v = 1; }
            for (size_t t = 0; t <= n; t++) {
                unsigned char b = t < n ? src[t] : 10;
                if (g_needkv) for (int a = g_kvptr[b]; a < g_kvptr[b + 1]; a++) g_kv[g_kvcol[a]] = g_kvval[a];
                int na = ns, from = 0;
                for (int k = 0; k < NB; k++) { from = na; na += apply(&L[k], b, cur, na, cur + na); }
                if (g_needkv) for (int a = g_kvptr[b]; a < g_kvptr[b + 1]; a++) g_kv[g_kvcol[a]] = 0;
                /* next state goes straight into the other buffer; label and event are read off */
                ns = 0; int l = 0, e = -1;
                for (int i = from; i < na; i++) {
                    int r = cur[i].c - lbase;
                    if (r < NSTATE) { nxt[ns].c = r; nxt[ns].v = cur[i].v; ns++; }
                    else if (r < lim) l = r - oL;
                    else e = r - oE;
                }
                if (t < n) lab[t] = (unsigned char)l;
                ev[t] = (signed char)e;
                Act *tmp = cur; cur = nxt; nxt = tmp;
            }
        }
        /* The pruning assumed |activation| <= bound. Every activation the fast kernel produced was
         * checked against it; if any broke it the pruning was unsound and the file is redone with
         * the reference kernel. */
        if (ref || g_vmax <= g_U) break;
        fprintf(stderr, "pylexnet: activation %g exceeds declared bound %g; using the reference kernel\n", (double)g_vmax, (double)g_U);
    }
    /* backward cell and head: r' = event, or the event's map of r, or r; class = label or r */
    int reg = 0;
    for (size_t t = n; t-- > 0;) {
        if (ev[t + 1] >= 0) reg = ev[t + 1] < 9 ? ev[t + 1] : evmap[(ev[t + 1] - 9) * 9 + reg];
        if (lab[t] == 9) lab[t] = (unsigned char)reg;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (bench) {
        double s = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
        fprintf(stderr, "%zu bytes, %.1f ns/byte, %.1f rows touched/byte (per block:", n, s * 1e9 / (double)n, (double)g_touched / (double)n);
        for (int k = 0; k < NB && k < 16; k++) fprintf(stderr, " %.1f", (double)g_tb[k] / (double)n);
        fprintf(stderr, ")\n");
        return 0;
    }
    if (classes) { fwrite(lab, 1, n, stdout); return 0; }
    int cur = 0;
    for (size_t t = 0; t < n; t++) {
        if (lab[t] != cur) { fputs("\x1b[0m", stdout); fputs(ANSI[lab[t]], stdout); cur = lab[t]; }
        putchar(src[t]);
    }
    fputs("\x1b[0m", stdout);
    return 0;
}
