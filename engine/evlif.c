// Event-driven leaky integrate-and-fire network, parameters of Shiu et al. 2024 (Nature 634:210).
//   v_rest = v_reset = -52 mV, v_th = -45 mV, tau_m = 20 ms, tau_syn = 5 ms,
//   refractory 2.2 ms, synaptic delay 1.8 ms, weight = w_scale * 0.275 mV * synapse_count * sign.
// Membrane state is stored as u = v - v_rest; a neuron sleeps when |u| and |g| drop below EPS
// and is woken by the next incoming spike, so the cost scales with activity, not with N.
//
// Graph file: int64 [N, E], int64 indptr[N+1], int32 post[E], float32 count*sign[E] (rows = presynaptic).
//
// Removed neurons (ev_set_removed) are taken out of the network completely: they are never
// integrated, never spike (also when they belong to a stimulus group) and receive nothing.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
static double now_ms(void) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
#define EXPORT __attribute__((visibility("default")))
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
#endif

#define V_REST (-52.0)
#define V_TH (-45.0)
#define TAU_M 20.0
#define TAU_S 5.0
#define T_REF 2.2
#define T_DLY 1.8
#define W_UNIT 0.275
#define EPS 1e-3f

typedef struct {
  const int32_t *idx;
  int32_t n;
  float rate_hz;
  float on_ms;
  float off_ms;  // <= 0: until the end of the run
} StimGroup;

typedef struct {
  int32_t *spike_count;     // [N] or NULL
  float *first_spike_ms;    // [N] or NULL, -1 = no spike
  float *readout_first_ms;  // [nread] or NULL
  int32_t *readout_count;   // [nread] or NULL
  float bin_ms;             // > 0 enables time bins
  int32_t nbins;
  int32_t *bin_spikes;      // [nbins] spikes of non-stimulated neurons
  int32_t *readout_bins;    // [nread * nbins]
  float bio_ms;
  float wall_ms;
  int32_t awake_after;      // neurons still awake at the end
  int32_t active;           // non-stimulated neurons with at least one spike
  int64_t total_spikes;     // including stimulated neurons
  int64_t syn_events;
  double mean_awake;
  int32_t quiet_exit;       // stopped early: network silent and no more input
  int32_t reserved;
} Result;

typedef struct { int32_t *a; int32_t n, cap; } vec;
static inline void vpush(vec *v, int32_t x) {
  if (v->n == v->cap) {
    v->cap = v->cap ? v->cap * 2 : 4096;
    v->a = (int32_t *)realloc(v->a, (size_t)v->cap * 4);
  }
  v->a[v->n++] = x;
}

typedef struct {
  int32_t N;
  int64_t E;
  int64_t *indptr;
  int32_t *post;
  float *w;    // scaled weights, mV
  float *w0;   // original count*sign, allocated on first ev_set_weights
  float scale; // w_scale * 0.275
  float *uth;  // threshold in u-space
  uint8_t *removed;
  int32_t *rpos;  // readout position of each neuron or -1
  float *u, *g;
  int32_t *refr;
  uint8_t *awake, *isstim;
  int32_t *alist;
  int32_t na;
  vec *ring;
  int32_t R;
  vec fired;
} Ctx;

static inline uint64_t splitmix(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static inline double urand(uint64_t *s) { return ((splitmix(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

EXPORT void ev_free(void *p) {
  Ctx *c = (Ctx *)p;
  if (!c) return;
  free(c->indptr); free(c->post); free(c->w); free(c->w0); free(c->uth); free(c->removed); free(c->rpos);
  free(c->u); free(c->g); free(c->refr); free(c->awake); free(c->isstim); free(c->alist);
  if (c->ring) {
    for (int r = 0; r < c->R; r++) free(c->ring[r].a);
    free(c->ring);
  }
  free(c->fired.a);
  free(c);
}

EXPORT void *ev_load(const char *path, float w_scale) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return NULL; }
  int64_t hdr[2];
  if (fread(hdr, 8, 2, f) != 2) { fclose(f); return NULL; }
  Ctx *c = (Ctx *)calloc(1, sizeof(Ctx));
  c->N = (int32_t)hdr[0];
  c->E = hdr[1];
  c->indptr = (int64_t *)malloc((size_t)(c->N + 1) * 8);
  c->post = (int32_t *)malloc((size_t)c->E * 4 + 4);
  c->w = (float *)malloc((size_t)c->E * 4 + 4);
  if (fread(c->indptr, 8, c->N + 1, f) != (size_t)(c->N + 1) || fread(c->post, 4, c->E, f) != (size_t)c->E ||
      fread(c->w, 4, c->E, f) != (size_t)c->E) {
    fclose(f); ev_free(c); return NULL;
  }
  fclose(f);
  c->scale = (float)(w_scale * W_UNIT);
  for (int64_t e = 0; e < c->E; e++) c->w[e] *= c->scale;
  int N = c->N;
  c->uth = (float *)malloc((size_t)N * 4);
  for (int i = 0; i < N; i++) c->uth[i] = (float)(V_TH - V_REST);
  c->removed = (uint8_t *)calloc(N, 1);
  c->rpos = (int32_t *)malloc((size_t)N * 4);
  for (int i = 0; i < N; i++) c->rpos[i] = -1;
  c->u = (float *)calloc(N, 4);
  c->g = (float *)calloc(N, 4);
  c->refr = (int32_t *)malloc((size_t)N * 4);
  for (int i = 0; i < N; i++) c->refr[i] = INT32_MIN;
  c->awake = (uint8_t *)calloc(N, 1);
  c->isstim = (uint8_t *)calloc(N, 1);
  c->alist = (int32_t *)malloc((size_t)N * 4);
  return c;
}

EXPORT int32_t ev_n(void *p) { return ((Ctx *)p)->N; }
EXPORT int64_t ev_nnz(void *p) { return ((Ctx *)p)->E; }

EXPORT void ev_reset(void *p) {
  Ctx *c = (Ctx *)p;
  memset(c->u, 0, (size_t)c->N * 4);
  memset(c->g, 0, (size_t)c->N * 4);
  memset(c->awake, 0, c->N);
  for (int i = 0; i < c->N; i++) c->refr[i] = INT32_MIN;
  c->na = 0;
  if (c->ring)
    for (int r = 0; r < c->R; r++) c->ring[r].n = 0;
  c->fired.n = 0;
}

// Per-neuron threshold in mV (NULL = -45 mV for everyone).
EXPORT void ev_set_vth(void *p, const float *vth) {
  Ctx *c = (Ctx *)p;
  for (int i = 0; i < c->N; i++) c->uth[i] = (float)((vth ? vth[i] : V_TH) - V_REST);
}

// removed[i] = 1 takes neuron i out of the network (NULL = nobody removed). Resets the state.
EXPORT void ev_set_removed(void *p, const uint8_t *removed) {
  Ctx *c = (Ctx *)p;
  if (removed) memcpy(c->removed, removed, c->N);
  else memset(c->removed, 0, c->N);
  ev_reset(p);
}

// Replace edge weights with new count*sign values (same edge order as the file); NULL restores the file.
EXPORT void ev_set_weights(void *p, const float *counts) {
  Ctx *c = (Ctx *)p;
  if (!c->w0) {
    c->w0 = (float *)malloc((size_t)c->E * 4 + 4);
    for (int64_t e = 0; e < c->E; e++) c->w0[e] = c->w[e] / c->scale;
  }
  const float *src = counts ? counts : c->w0;
  for (int64_t e = 0; e < c->E; e++) c->w[e] = src[e] * c->scale;
  ev_reset(p);
}

EXPORT int ev_run(void *p, const StimGroup *groups, int32_t ngroups, float dt_ms, float max_ms, uint64_t seed,
                  const int32_t *readouts, int32_t nread, Result *out) {
  Ctx *c = (Ctx *)p;
  const int N = c->N;
  const double dt = dt_ms;
  const float em = (float)exp(-dt / TAU_M), es = (float)exp(-dt / TAU_S);
  const float kk = (float)((TAU_S / (TAU_S - TAU_M)) * (exp(-dt / TAU_S) - exp(-dt / TAU_M)));
  const int D = (int)lround(T_DLY / dt), RF = (int)lround(T_REF / dt);
  const int R = D + 1;
  const long steps = lround(max_ms / dt);
  if (c->R != R) {
    if (c->ring) {
      for (int r = 0; r < c->R; r++) free(c->ring[r].a);
      free(c->ring);
    }
    c->ring = (vec *)calloc(R, sizeof(vec));
    c->R = R;
  }
  uint64_t rs = seed ^ 0xD1B54A32D192ED03ULL;

  int64_t ns = 0;
  for (int gi = 0; gi < ngroups; gi++) ns += groups[gi].n;
  int32_t *sidx = (int32_t *)malloc((size_t)(ns + 1) * 4);
  int32_t *sgrp = (int32_t *)malloc((size_t)(ns + 1) * 4);
  int64_t *next = (int64_t *)malloc((size_t)(ns + 1) * 8);
  double *lp = (double *)malloc((size_t)(ngroups + 1) * sizeof(double));
  long *on = (long *)malloc((size_t)(ngroups + 1) * sizeof(long));
  long *off = (long *)malloc((size_t)(ngroups + 1) * sizeof(long));
  memset(c->isstim, 0, N);
  int64_t k = 0;
  long last_on = 0;
  for (int gi = 0; gi < ngroups; gi++) {
    double pr = groups[gi].rate_hz * dt / 1000.0;
    if (pr > 0.999) pr = 0.999;
    lp[gi] = pr > 0 ? log(1.0 - pr) : 0;
    on[gi] = lround(groups[gi].on_ms / dt);
    off[gi] = groups[gi].off_ms > 0 ? lround(groups[gi].off_ms / dt) : steps;
    if (pr > 0 && off[gi] > last_on) last_on = off[gi];
    for (int j = 0; j < groups[gi].n; j++) {
      int32_t i = groups[gi].idx[j];
      if (i < 0 || i >= N || c->removed[i]) continue;
      sidx[k] = i;
      sgrp[k] = gi;
      // stimulated neurons follow their Poisson source and do not integrate input
      if (pr > 0 && off[gi] > on[gi]) {
        next[k] = on[gi] + (int64_t)floor(log(urand(&rs)) / lp[gi]);
        c->isstim[i] = 1;
      } else next[k] = INT64_MAX;
      k++;
    }
  }
  ns = k;
  {
    int w2 = 0;
    for (int a = 0; a < c->na; a++) {
      int i = c->alist[a];
      if (c->isstim[i]) c->awake[i] = 0;
      else c->alist[w2++] = i;
    }
    c->na = w2;
  }

  if (out->spike_count) memset(out->spike_count, 0, (size_t)N * 4);
  if (out->first_spike_ms)
    for (int i = 0; i < N; i++) out->first_spike_ms[i] = -1.f;
  if (out->readout_first_ms)
    for (int r = 0; r < nread; r++) out->readout_first_ms[r] = -1.f;
  if (out->readout_count) memset(out->readout_count, 0, (size_t)nread * 4);
  const int nb = (out->bin_ms > 0 && out->nbins > 0) ? out->nbins : 0;
  if (nb && out->bin_spikes) memset(out->bin_spikes, 0, (size_t)nb * 4);
  if (nb && out->readout_bins) memset(out->readout_bins, 0, (size_t)nb * nread * 4);
  for (int r = 0; r < nread; r++)
    if (readouts[r] >= 0 && readouts[r] < N) c->rpos[readouts[r]] = r;
  int32_t *rcnt = (int32_t *)calloc(nread + 1, 4);
  uint8_t *seen = (uint8_t *)calloc(N, 1);
  int32_t active = 0;
  out->quiet_exit = 0;
  int64_t tot_spk = 0, tot_ev = 0;
  double sum_awake = 0;

  const double t0 = now_ms();
  long t = 0;
  int32_t *alist = c->alist;
  float *u = c->u, *g = c->g;
  int32_t *refr = c->refr;
  uint8_t *awake = c->awake;
  const uint8_t *isstim = c->isstim, *removed = c->removed;
  const float *uth = c->uth;
  const int32_t *rpos = c->rpos;
  for (t = 0; t < steps; t++) {
    c->fired.n = 0;
    int na = c->na, w2 = 0;
    for (int a = 0; a < na; a++) {
      int i = alist[a];
      if (refr[i] > t) { alist[w2++] = i; continue; }
      float uu = u[i] * em + g[i] * kk;
      float gg = g[i] * es;
      u[i] = uu;
      g[i] = gg;
      if (uu > uth[i]) { vpush(&c->fired, i); alist[w2++] = i; continue; }
      if (fabsf(uu) < EPS && fabsf(gg) < EPS) { u[i] = 0; g[i] = 0; awake[i] = 0; continue; }
      alist[w2++] = i;
    }
    c->na = na = w2;
    sum_awake += na;
    for (int64_t s = 0; s < ns; s++) {
      int gi = sgrp[s];
      while (next[s] <= t) {
        if (next[s] == t && t < off[gi]) vpush(&c->fired, sidx[s]);
        if (t >= off[gi]) { next[s] = INT64_MAX; break; }
        next[s] += 1 + (int64_t)floor(log(urand(&rs)) / lp[gi]);
      }
    }
    vec *slot = &c->ring[t % R];
    for (int q = 0; q < slot->n; q++) {
      int pre = slot->a[q];
      int64_t e0 = c->indptr[pre], e1 = c->indptr[pre + 1];
      tot_ev += e1 - e0;
      for (int64_t e = e0; e < e1; e++) {
        int post = c->post[e];
        if (removed[post]) continue;
        g[post] += c->w[e];
        if (!awake[post] && !isstim[post]) { awake[post] = 1; alist[c->na++] = post; }
      }
    }
    slot->n = 0;
    vec *o = &c->ring[(t + D) % R];
    const float tms = (float)(t * dt);  // spike time = start of the step
    const int bi = nb ? (int)(t * dt / out->bin_ms) : 0;
    for (int q = 0; q < c->fired.n; q++) {
      int i = c->fired.a[q];
      u[i] = 0;
      g[i] = 0;
      refr[i] = (int32_t)(t + RF);
      if (out->spike_count) out->spike_count[i]++;
      if (out->first_spike_ms && out->first_spike_ms[i] < 0) out->first_spike_ms[i] = tms;
      vpush(o, i);
      if (!isstim[i]) {
        if (!seen[i]) { seen[i] = 1; active++; }
        if (nb && bi < nb && out->bin_spikes) out->bin_spikes[bi]++;
      }
      int r = rpos[i];
      if (r >= 0) {
        rcnt[r]++;
        if (out->readout_first_ms && out->readout_first_ms[r] < 0) out->readout_first_ms[r] = tms;
        if (nb && bi < nb && out->readout_bins) out->readout_bins[r * nb + bi]++;
      }
    }
    tot_spk += c->fired.n;
    if (c->na == 0 && t >= last_on) {
      int pend = 0;
      for (int r = 0; r < R; r++) pend += c->ring[r].n;
      if (!pend) { out->quiet_exit = 1; t++; break; }
    }
  }
  out->wall_ms = (float)(now_ms() - t0);
  out->bio_ms = (float)(t * dt);
  out->total_spikes = tot_spk;
  out->syn_events = tot_ev;
  out->mean_awake = t ? sum_awake / t : 0;
  out->awake_after = c->na;
  out->active = active;
  if (out->readout_count)
    for (int r = 0; r < nread; r++) out->readout_count[r] = rcnt[r];
  for (int r = 0; r < nread; r++)
    if (readouts[r] >= 0 && readouts[r] < N) c->rpos[readouts[r]] = -1;
  for (int64_t s2 = 0; s2 < ns; s2++) {
    int i = sidx[s2];
    if (c->isstim[i]) { u[i] = 0; g[i] = 0; }
  }
  memset(c->isstim, 0, N);
  free(seen); free(rcnt); free(sidx); free(sgrp); free(next); free(lp); free(on); free(off);
  return 0;
}
