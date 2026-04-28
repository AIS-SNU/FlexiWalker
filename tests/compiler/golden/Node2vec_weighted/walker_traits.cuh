// Auto-generated WalkerTraits specializations based on class metadata

#include "app.cuh"

template <typename T>
struct WalkerTraits;

template <>
struct WalkerTraits<Deepwalk> {
    static constexpr bool POSSIBLE_ZERO = 0;
    static constexpr char UPDATE_FLAG = 1;
    static constexpr char ERVS_ONLY = 0;
};

template <>
struct WalkerTraits<Metapath> {
    static constexpr bool POSSIBLE_ZERO = 1;
    static constexpr char UPDATE_FLAG = 0;
    static constexpr char ERVS_ONLY = 0;
};

template <>
struct WalkerTraits<Metapath_weighted> {
    static constexpr bool POSSIBLE_ZERO = 1;
    static constexpr char UPDATE_FLAG = 1;
    static constexpr char ERVS_ONLY = 0;
};

template <>
struct WalkerTraits<Node2vec> {
    static constexpr bool POSSIBLE_ZERO = 0;
    static constexpr char UPDATE_FLAG = 0;
    static constexpr char ERVS_ONLY = 0;
};

template <>
struct WalkerTraits<Node2vec_weighted> {
    static constexpr bool POSSIBLE_ZERO = 0;
    static constexpr char UPDATE_FLAG = 1;
    static constexpr char ERVS_ONLY = 0;
};

template <>
struct WalkerTraits<PPR> {
    static constexpr bool POSSIBLE_ZERO = 0;
    static constexpr char UPDATE_FLAG = 1;
    static constexpr char ERVS_ONLY = 0;
};

template <>
struct WalkerTraits<PPR_second> {
    static constexpr bool POSSIBLE_ZERO = 0;
    static constexpr char UPDATE_FLAG = 1;
    static constexpr char ERVS_ONLY = 0;
};

