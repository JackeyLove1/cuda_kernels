#include <cstdio>
#include <iostream>

#define INLINE __attribute__((always_inline))

template <auto v>
struct C
{
    using type = C<v>;
    static constexpr auto value = v;
    using value_type = decltype(value);
};



int main()
{
    std::cout << "Hello world!" << std::endl;
    std::cout << C<1>::value << std::endl;
    return 0;
}