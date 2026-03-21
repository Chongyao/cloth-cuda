#pragma once
#include <cstdio>
static int g_passed = 0, g_failed = 0;
#define CHECK(cond) \
    do { if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
        ++g_failed; \
    } else { ++g_passed; } } while(0)
#define CHECK_EQ(a, b)  CHECK((a) == (b))
#define SECTION(name)   printf("-- %s\n", (name))
inline int test_summary() {
    printf("Results: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
