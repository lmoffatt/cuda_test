#ifndef STATIC_STRING_H
#define STATIC_STRING_H

#include <string>
#include <iomanip>
#include <cstdio>
#include "/usr/local/cuda-11.0/targets/x86_64-linux/include/cuda_runtime.h"


template <std::size_t N> struct my_static_string {
private:
  std::array<char, N> c_;
public:
  constexpr my_static_string(const char (&c)[N]) : c_{} {
    for (std::size_t i = 0; i < N; ++i)
      c_[i] = c[i];
  }
  constexpr my_static_string(std::array<char, N> c) : c_{c} {
  }
  constexpr char operator[](std::size_t i) const { return c_[i]; }

  template <std::size_t N0>
  constexpr my_static_string(my_static_string<N0> one,
                             my_static_string<N - N0 + 1> two)
      : c_{} {
    for (std::size_t i = 0; i < N0 - 1; ++i)
      c_[i] = one[i];
    for (std::size_t i = N0 - 1; i < N - 1; ++i)
      c_[i] = two[i + 1 - N0];

    c_[N - 1] = '\0';
  }

  constexpr const char * c_str() const { return &c_[0]; }

  friend
      __host__ __device__
      void my_print(const  my_static_string& me)
  {
    for (std::size_t i = 0; i < N-1; ++i)
      printf("%c", me[i]);

  }



  friend
      std::istream& operator>>(std::istream& is, my_static_string&& m)
  {
    for (auto i=0u; i<N-1; ++i)
    {
      auto c=is.get();
      if (c!=m.c_[i])    {
        is.setstate(std::ios::failbit);
        return is;
      }

    }
    return is;
  }
  friend
      std::istream& operator>>(std::istream& is, my_static_string& m)
  {
    for (auto i=0u; i<N; ++i)
    {
      m.c_[i]=is.get();
    }
    return is;
  }

  std::string str() const { return c_str(); }

    friend constexpr  bool operator == (const my_static_string& one, const my_static_string& two )
  {
    for (std::size_t i=0; i<N; ++i)
    {
      if(one[i]!=two[i]) return false;
    }
    return true;
  }
  template <std::size_t N2>
  friend constexpr  bool operator == (const my_static_string& , const my_static_string<N2>&  )
  {return false;}

};

  template <std::size_t N,std::size_t N2>
 constexpr  bool operator < (const my_static_string<N>& one, const my_static_string<N2>& two )
{
  for (std::size_t i=0; i<std::min(N,N2); ++i)
  {
    if(one[i]<two[i]) return true;
    else if (one[i]>two[i]) return false;
  }
  return N<N2;
}


template <int N>
my_static_string(const char (&lit)[N])  ->my_static_string<N>;

template <std::size_t N1, std::size_t N2>
constexpr auto operator+(const my_static_string<N1> &s1,
                         const my_static_string<N2> &s2)
    -> my_static_string<N1 + N2 - 1> {
  return my_static_string<N1 + N2 - 1>(s1, s2);
}

template <int N>
constexpr auto to_static_string()
{
  if constexpr (N<0)
  {
    return my_static_string("-")+to_static_string<-N>();
  }
  else if constexpr (N<10)
  {
    constexpr char c='0'+N;
    return my_static_string({c,'\0'});
  }
  else
  {
    constexpr int N10=N/10;
    constexpr int N0=N-N10*10;
    return to_static_string<N10>()+ my_static_string<1>('0' + N0)+my_static_string<1>('\0');
  }
}

#endif // STATIC_STRING_H
