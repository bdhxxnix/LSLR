// Isolated compilation unit for DynamicLearnedIndex baseline.
#include "baseline_wrapper.hpp"

#include "big_quotient.h"
#define STORAGE true
#define OPT false
#include "learned_index/LearnedIndex.h"

#include <stack>
#include <algorithm>
#include <cmath>

// ---- Template implementation per epsilon ----
template<int Epsilon>
struct ImplT {
    LearnedIndex<int64_t, Epsilon, BigQuotient> index;
};

namespace {

template<int Epsilon>
bool do_insert(void* impl, int64_t key) {
    static_cast<ImplT<Epsilon>*>(impl)->index.insert(key);
    return true;
}

template<int Epsilon>
bool do_find(void* impl, int64_t key) {
    return static_cast<ImplT<Epsilon>*>(impl)->index.find(key);
}

template<int Epsilon>
int do_segment_count(void* impl) {
    return static_cast<ImplT<Epsilon>*>(impl)->index.segments_count();
}

template<int Epsilon>
size_t do_memory_bytes(void* impl) {
    return static_cast<ImplT<Epsilon>*>(impl)->index.size_in_bytes();
}

template<int Epsilon>
std::vector<std::pair<int64_t, int64_t>> do_segment_ranges(void* impl) {
    auto* p = static_cast<ImplT<Epsilon>*>(impl);
    std::vector<std::pair<int64_t, int64_t>> result;
    auto root = p->index.lineTree.root;
    if (!root) return result;

    std::stack<decltype(root)> stk;
    stk.push(root);
    while (!stk.empty()) {
        auto x = stk.top();
        stk.pop();
        if (!x->left && !x->right) {
            result.emplace_back(x->val.s, x->val.t);
        } else {
            if (x->right) stk.push(x->right);
            if (x->left) stk.push(x->left);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

struct VTable {
    bool (*insert)(void*, int64_t);
    bool (*find)(void*, int64_t);
    int  (*segment_count)(void*);
    size_t (*memory_bytes)(void*);
    std::vector<std::pair<int64_t, int64_t>> (*segment_ranges)(void*);
};

template<int Epsilon>
VTable make_vtable() {
    return { &do_insert<Epsilon>, &do_find<Epsilon>,
             &do_segment_count<Epsilon>, &do_memory_bytes<Epsilon>,
             &do_segment_ranges<Epsilon> };
}

template<int Epsilon>
void* create_impl() { return new ImplT<Epsilon>(); }

template<int Epsilon>
void destroy_impl(void* impl) { delete static_cast<ImplT<Epsilon>*>(impl); }

using CreateFn  = void* (*)();
using DestroyFn = void (*)(void*);

struct EpsilonEntry {
    int epsilon;
    VTable vtable;
    CreateFn create;
    DestroyFn destroy;
};

#define REG(EPS) { EPS, make_vtable<EPS>(), create_impl<EPS>, destroy_impl<EPS> }

const EpsilonEntry epsilon_registry[] = {
    REG(64),
    REG(500),
    REG(1000),
    REG(2000),
    REG(5000),
    REG(10000),
    REG(15000),
    REG(20000),
    REG(25000),
    REG(30000),
    REG(40000),
    REG(50000),
};

#undef REG

const EpsilonEntry* find_entry(int epsilon) {
    const EpsilonEntry* best = &epsilon_registry[0];
    int best_diff = std::abs(epsilon - best->epsilon);
    for (const auto& entry : epsilon_registry) {
        int diff = std::abs(epsilon - entry.epsilon);
        if (diff < best_diff) { best = &entry; best_diff = diff; }
    }
    return best;
}

} // anonymous namespace

struct BaselineIndex::Impl {
    void* data = nullptr;
    VTable vtable;
    DestroyFn destroy = nullptr;
};

BaselineIndex::BaselineIndex(int epsilon) : epsilon_(epsilon) {
    auto* entry = find_entry(epsilon);
    impl_ = new Impl();
    impl_->data = entry->create();
    impl_->vtable = entry->vtable;
    impl_->destroy = entry->destroy;
}

BaselineIndex::~BaselineIndex() {
    if (impl_) {
        if (impl_->data && impl_->destroy) impl_->destroy(impl_->data);
        delete impl_;
    }
}

bool BaselineIndex::insert(int64_t key) { return impl_->vtable.insert(impl_->data, key); }
bool BaselineIndex::find(int64_t key)   { return impl_->vtable.find(impl_->data, key); }
int  BaselineIndex::segment_count() const { return impl_->vtable.segment_count(impl_->data); }
size_t BaselineIndex::memory_bytes() const { return impl_->vtable.memory_bytes(impl_->data); }
std::vector<std::pair<int64_t, int64_t>> BaselineIndex::segment_ranges() const {
    return impl_->vtable.segment_ranges(impl_->data);
}
