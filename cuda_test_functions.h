#ifndef CUDA_TEST_FUNCTIONS_H
#define CUDA_TEST_FUNCTIONS_H


#include "static_string.h"

#include <type_traits>
#include <utility>
#include <vector>
#include <cstdio>

__host__ __device__
    void my_print(double x){ printf("%g",x);}

__host__ __device__
    void my_print(const char* x){ printf("%s",x);}



template<template<class...>class, class>
struct is_this_template_class
{
  static constexpr bool value=false;
};



template<template<class ...>class V, class...Ts>
struct is_this_template_class<V,V<Ts...>>
{

  static constexpr bool value=true;
};

template<template<class ...>class V, class T>
inline constexpr bool is_this_template_class_v= is_this_template_class<V,std::decay_t<T>>::value;




template<template<class ...>class... Vs>
struct is_any_of_these_template_classes
{
  template<class T>
  static constexpr bool value=(is_this_template_class_v<Vs,T>||...);
};


template<class T>
class vector_device
{
private:
  T* v_=nullptr;
  std::size_t n_=0;


public:
  __host__ __device__
      auto& operator[](std::size_t i){return *(v_+i);}
  __host__ __device__
      auto& operator[](std::size_t i)const {return *(v_+i);}

  __host__ __device__
      auto& begin(){return v_;}

  __host__ __device__
      auto const begin()const {return v_;}

  __host__ __device__
      auto end(){return v_+n_;}

  __host__ __device__
      auto const  end()const {return v_+n_;}

  __host__ __device__
  vector_device(const vector_device& other):v_{new T[other.size()]},n_{other.size()}
  {
    for (std::size_t i=0; i<size(); ++i)
      (*this)[i]=other[i];
  }

  __host__ __device__
      explicit vector_device(vector_device&& other):v_{other.begin()},n_{other.size()}
  {
    other.v_=nullptr;
    other.n_=0;
  }

  // __host__ __device__
  vector_device(std::size_t n):
                                 v_{new T[n]},n_{n}{}

  __host__ __device__
  vector_device(std::size_t n, T x):v_{new T[n]},n_{n}
  {
    for (std::size_t i=0; i<size(); ++i)
      (*this)[i]=x;
  }


  __host__ __device__
      vector_device& operator=(const vector_device& other)const
  {
    if (other.size()!=size())
    {  delete[] v_; n_=other.size(); v_=new T[n_];}

    for (std::size_t i=0; i<size(); ++i)
      (*this)[i]=other[i];
    return *this;
  }


  __host__ __device__
      vector_device& operator=(vector_device&& other)const
  {
    (*this)=other;
    other.v_=nullptr;
    other.n_=0;
  }

  __host__ __device__
      friend void swap (vector_device& one, vector_device & two)
  {
    T* tmp=one.v_;
    one.v_=two.v_;
    two.v_=tmp;
    auto n=one.n_;
    one.n_=two.n_;
    two.n_=n;
  }



 // __host__ __device__
  vector_device()=default;

  __host__ __device__
      ~vector_device()
  {
    delete[] v_;
  }

#ifndef __CUDA_ARCH__
//host code here
#else
//device code here
#endif

  __host__ __device__
      constexpr  auto size()const{return n_;}
  __host__ __device__
      constexpr  auto byte_size()const {return size()*sizeof (T);}


  friend
  __host__ __device__
  void my_print(const vector_device& me)
  {
    printf("{");
    for (std::size_t i=0; i<me.size(); ++i)
      my_print(me[i]);
    printf("}");

  }




  friend
      __host__
          std::ostream& operator<<(std::ostream& os, const vector_device& me)
  {
    os<<"{";
    for (std::size_t i=0; i<me.size(); ++i)
      os<<me[i];
    return os<<"}";
  }


};






template<class T, class S>
auto operator*(const vector_device<T>& x, const S& y)
{
  vector_device<std::decay_t<decltype (std::declval<T>()*y)> > out(x.size());
  for (std::size_t i=0; i<out.size(); ++i)
    out[i]=x[i]*y;
  return out;
}

template<class T, class S>
auto operator*(const S& y, const vector_device<T>& x)
{
  vector_device<std::decay_t<decltype (y*std::declval<T>())> > out(x.size());
  for (std::size_t i=0; i<out.size(); ++i)
    out[i]=y*x[i];
  return out;
}


template<class...> struct Cs{};



struct Nothing{
  template<class Something>
  friend
      __host__ __device__
      decltype(auto) operator+( Something&& s, Nothing)//->decltype (s)
  {return std::forward<Something>(s);}

  template<class Something>
  friend
      __host__ __device__
      decltype(auto) operator+( Nothing, Something&& s)//->decltype (s)
  {return std::forward<Something>(s);}

  friend
      __host__ __device__
      Nothing operator +(Nothing, Nothing){return Nothing{};}

  __host__ __device__
  Nothing operator()()const{return Nothing{};}

  template<class Position>
  __host__ __device__
  Nothing operator()(Position)const{return Nothing{};}

};

constexpr bool is_Nothing(...){return false;}

constexpr bool is_Nothing(Nothing){return true;}


template <template<class...> class V0,template<class...> class V1,class ...v0,class...v1>
auto transfer(V0<v0...>,V1<v1...>){return V1<v0...,v1...>{};}


template<class T>
class First_Result{
  T result_;
public:
  __host__ __device__
  First_Result(T x): result_{x}{}
  friend
  __host__ __device__
  First_Result operator+( First_Result s, First_Result)//->decltype (s)
  {return s;}

  __host__ __device__
   T operator()(){return result_;}

};



  template<class T>
__host__ __device__
  First_Result<T> operator+(First_Result<T> one, First_Result<Nothing>)
  {
    return one;
  }
  template<class T>
  __host__ __device__
  First_Result<T> operator+(First_Result<Nothing>,First_Result<T> one)
  {
    return one;
  }



template <class...Args> struct Arguments{
  template<class ...Args2>
  friend constexpr auto  operator&&(Arguments,Arguments<Args2...>){return Arguments<Args...,Args2...>{};}

  static inline constexpr auto Argsname=my_static_string("args(");
  friend constexpr auto operator +(Arguments<>, Arguments){return Arguments{};}

  static inline constexpr auto name=Argsname+((Args::name+my_static_string(","))+...+my_static_string(")"));
  static inline constexpr auto size=sizeof... (Args);
  static   constexpr auto my_name(){return name;}

};


template <class...Ids> struct Variables{

  static inline constexpr auto size=sizeof... (Ids);

  static inline constexpr auto name=my_static_string("vars(")+((Ids::name+my_static_string(","))+...+my_static_string(")"));
  static   constexpr auto my_name(){return name;}




};

template<class... Ids,class... Id2>
constexpr auto operator&&(Variables<Ids...>,Variables<Id2...>){ return Variables<Ids...,Id2...>{};}

template<class...Ids>
constexpr auto operator-(Variables<>,Variables<Ids...>)
{
  return Variables<>{};
}
template<class...Ids,class  Id2>
constexpr auto operator-(Variables<Id2>,Variables<Ids...>)
{
  return std::conditional_t<(std::is_same_v<Ids,Id2 >||...),Variables<>,Variables<Id2>>{};
}

template<class Id,class Id1,class...Ids,class...  Id2>
constexpr auto operator-(Variables<Id,Id1,Ids...>,Variables<Id2...>)
{
  return (((Variables<Id>{}-Variables<Id2...>{})&&(Variables<Id1>{}-Variables<Id2...>{}))&&...&&(Variables<Ids>{}-Variables<Id2...>{}));
}

template<class... Id0, class... Id1>
constexpr auto operator-(Arguments<Id0...>,Variables<Id1...>)
{
  return transfer(Variables<Id0...>{}-Variables<Id1...>{},Arguments<>{});
}





template<class arg1, class arg2,class...args >
constexpr auto operator+(Arguments<arg1,args...>, Arguments<arg2>)
{
  if constexpr(std::is_same_v<arg1,arg2 >)
    return Arguments<arg1,args...>{};
  else if constexpr (arg2::name<arg1::name)
    return Arguments<arg2,arg1,args...>{};
  else
    return Arguments<arg1>{}&&(Arguments<args...>{}+Arguments<arg2>{});

}



template <class F, class X>
struct Not_found{
  static inline constexpr auto not_found_name=my_static_string("not_found(");
  static inline constexpr auto name=not_found_name+F::name+my_static_string(": ")+X::name;
  Not_found(F,X){}

  Not_found(){}


};


template <class X>
struct Found
{
  X x;
  Found(X x_):x{std::forward<X>(x_)}{}
};




template <class...> struct Position;

template< >struct Position<>{};

template <class Id> struct Position<Id>{
private :
  std::size_t i;
public:
  static constexpr auto pos_name=my_static_string("pos_");
  static constexpr auto name=pos_name+Id::name;
  __device__ __host__ Position(std::size_t i_v):
                                                  i{i_v}{}
  Position()=default;
  using innerId=Id;
  __host__ __device__
 auto& operator[](Id)const {return *this;}
  __host__ __device__
  auto& operator[](Id) {return *this;}
  __host__ __device__
  auto& operator()()const {return  i;}
  __device__ __host__ auto& operator()() {return  i;}
  auto& operator++(){ ++i; return *this;}
  __host__ __device__
  std::size_t inc(Id)
  {
    ++i;
    return i;
  }
  template<class... Idx>
  __host__ __device__
  Position(const Position<Idx...>& p):i{p[Id{}]()}{  }


  template<class ...Xs>
  __host__ __device__
  auto& operator()(const Position<Xs...>& ){return *this;}
  template<class ...Xs>
  __host__ __device__
  auto& operator()(const Position<Xs...>& )const {return *this;}

  friend
      __host__ __device__
      void my_print(const Position& me) {
    printf(name.c_str()),
        printf("=");
    my_print(me());}


  friend std::ostream& operator<<(std::ostream& os, const Position& me) {
    os<<name.c_str()<<"="<<me();
    return os;}
  friend std::istream& operator>>(std::istream& is, Position& me){ is>>name>>my_static_string("=")>>me(); return is;}

};


template <class Id, class... Ids> struct Position<Id,Ids...>:Position<Id>,Position<Ids>...
{
  static constexpr auto name=(Position<Id>::name+...+Position<Ids>::className);


  template<class aPosition>
  __host__ __device__
  Position (const aPosition& p):Position<Id>{p},Position<Ids>{p}...{}

  Position()=default;

  template<class ...Xs>
  __host__ __device__
  auto& operator()(const Position<Xs...>& ) {return *this;}

  template<class ...Xs>
  __host__ __device__
  auto& operator()(const Position<Xs...>& )const {return *this;}

  using Position<Id>::operator[];
  using Position<Ids>::operator[]...;

  friend
      __host__ __device__
      void my_print( const Position& me) {
    my_print(me[Id{}]);
    if constexpr (sizeof... (Ids)>0)
      ((my_print("_"),my_print(me[Ids{}])),...);
  }


  friend std::ostream& operator<<(std::ostream& os, const Position& me) {
    os<<me[Id{}];
    if constexpr (sizeof... (Ids)>0)
      ((os<<"_"<<me[Ids{}]),...);
    return os;
  }

};







template<class Value_type,class Idx> class vector_field
{
public:
  typedef Value_type element_type;

  typedef vector_device<element_type> value_type;


private:
  value_type value_;

public:



  __host__ __device__
      explicit vector_field(vector_device<element_type>&& x):value_{std::move(x)}{}

  __host__ __device__
      auto& value(){return value_;}


  __host__ __device__
      auto& value()const {return value_;}

  //__host__ __device__
  vector_field()=default;

  __host__ __device__
      auto size()const{return value().size();}

  auto byte_size()const {return value().byte_size();;}

  template<class...Xs>
  __host__ __device__
      decltype(auto) operator()(const Position<Xs...>& p)const { return value()[p[Idx{}]()];}

  template<class...Xs>
  __host__ __device__
      decltype(auto) operator()(const Position<Xs...>& p) { return value()[p[Idx{}]()];}


  friend
      __host__ __device__
  void my_print(const vector_field& me)
  {
    printf("{");
    for (auto& e:me.value()) { my_print(e);printf(", ");}
    printf("}");
  }



  friend
      __host__
          std::ostream& operator<<(std::ostream& os, const vector_field& me)
  {
    os<<"{";
    for (auto& e:me.value()) os<<e<<", ";
    return os<<"}";
  }

};


template<class Id, typename T>
class x_i
{

private:
  T x_;

public:
  inline static constexpr auto x_iname=my_static_string("x_i(");
  using myId=Id;

  __host__ __device__
      constexpr x_i(Id, T&& x):x_{std::move(x)}{}

  __host__ __device__
      constexpr x_i(Id, T const& x):x_{x}{}

  template<class U>
  __host__ __device__
      constexpr  x_i(Id, vector_device<U> x):x_{std::move(x)}{}


  __host__ __device__
      constexpr auto& operator[](Id)const {return *this;}
  __host__ __device__
      constexpr auto& operator[](Id) {return *this;}
  __host__ __device__
      constexpr auto& operator()()const {return x_;}
  __host__ __device__
      constexpr auto& operator()() {return x_;}


  friend
      __host__ __device__
      void my_print(const x_i& me)
  {
    my_print("x_i(");
    my_print(Id::my_name());
    my_print("=");
    my_print(me());
    my_print(")");

  }



  friend
      std::ostream& operator<<(std::ostream& os, const x_i& me)
  {
    return os<<x_iname.str()<<Id::name.str()<<"="<<me()<<")";

  }



};

template<class Id, typename T>
x_i(Id,T&&)->x_i<Id,T>;
template<class Id, typename T>
x_i(Id,T const&)->x_i<Id,T>;

template<class Id, typename T>
x_i(Id,vector_device<T>&&)->x_i<Id,vector_field<T,Id>>;

template<class Id, typename T>
x_i(Id,vector_device<T>const &)->x_i<Id,vector_field<T,Id>>;



template <class...xs>
struct mapu: public xs...
{
  using xs::operator[]...;

  constexpr inline static auto size=sizeof... (xs);


  template<class Id>
  __host__ __device__
  Nothing operator[](Id)const {return Nothing{};}


  __host__ __device__
  constexpr mapu(xs...x):xs{x}...{}


  template<class ... xs2>
  __host__ __device__
  friend auto operator&&(mapu&& x, mapu<xs2...>&& y)
  {
    return mapu<xs...,xs2...>(std::move(x)[typename xs::myId{}]...,std::move(y)[typename xs2::myId{}]...);
  }

  friend
      __host__ __device__
      void my_print(const mapu& me)
  {
    ((my_print(me[typename xs::myId{}]),my_print("\n")),...);
  }


  friend std::ostream& operator<<(std::ostream& os, const mapu& me)
  {
    return ((os<<me[typename xs::myId{}]<<"\n"),...,os);
  }

};








template <class...xs>
struct quimulun: public xs...
{
  static inline constexpr auto quimulun_name=my_static_string("quimulun(");
  static inline constexpr auto name=(quimulun_name+...+xs::name)+my_static_string(")");
  using xs::operator[]...;

  constexpr quimulun(xs...x):xs{x}...{}

  constexpr quimulun(){}
};
template <class...xs>
struct Error
{
  static inline constexpr auto Errorname=my_static_string("Error(");

  static inline constexpr auto name=(Errorname+...+xs::name)+my_static_string(")");

  template<class... xs2>
  friend auto operator&&(Error<xs...>, Error<xs2...>){return Error<xs...,xs2...>{};}

  template<class x,class... xs2>
  friend decltype(auto) operator&&(mapu<x,xs2...>&& out, Error){return std::move(out);}

  template<class...xs2>
  friend decltype(auto) operator&&(Error,mapu<xs2...>&& out){return std::move(out);}

  Error(xs...){}
  Error(){}

  friend
      __host__ __device__
      void my_print(std::ostream& os, Error){my_print(name.c_str());}


  friend std::ostream& operator<<(std::ostream& os, Error){os<<name.str(); return os;}
};

template<class> struct is_Error:public std::false_type{};

template <> struct is_Error<Nothing>:public std::true_type{};



template <class...xs> struct is_Error<Error<xs...>>:public std::true_type {};

template<class T> inline constexpr auto is_Error_v=is_Error<std::decay_t<T>>::value;

template<class Id,class T>
auto variables(x_i<Id,T>) {

  return Variables<Id>{};

}

template <class...xs>
auto variables(quimulun<xs...>)
{
  return decltype ((variables(std::declval<xs>())&&...)){};
}
template <class...xs>
auto variables(const mapu<xs...>&)
{
  //  using test=typename Cs<xs...,
  //                           decltype ((variables(std::declval<xs>())&&...&&Variables<>{})),
  //                           decltype (variables(std::declval<xs>()))...>::var;
  return decltype ((variables(std::declval<xs>())&&...&&Variables<>{})){};
}






template <int v>
struct N
{inline static constexpr auto is_identifier=true;
  inline static constexpr auto name=to_static_string<v>();

};



struct S{
  inline static constexpr auto name=my_static_string("sum");
  template<class X, class Y>
  inline static constexpr auto get_name(){return X::name+my_static_string("+")+Y::name;}
  inline static constexpr auto get_name(){return name;}
};
struct P{
  inline static constexpr auto name=my_static_string("product");
  template<class X, class Y>
  inline static constexpr auto get_name(){return X::name+my_static_string("*")+Y::name;}
};

struct Index{
  inline static constexpr auto Index_name=my_static_string("index(");

  template<class Max>
  inline static constexpr auto get_name(){return Index_name+Max::name+my_static_string(")");}
  template<class Min,class Max>
  inline static constexpr auto get_name(){return Index_name+Min::name+my_static_string(",")+Max::name+my_static_string(")");}

};

struct DeRef{
  inline static constexpr auto evalname=my_static_string("deref(");
  template<class... X>
  inline static constexpr auto get_name(){return ((evalname+...+X::name)+my_static_string(")"));}
};


struct Eval{
  inline static constexpr auto evalname=my_static_string("eval(");
  template<class... X>
  inline static constexpr auto get_name(){return ((evalname+...+X::name)+my_static_string(")"));}
};

struct Reserve{
  inline static constexpr auto evalname=my_static_string("reserve(");
  template<class... X>
  inline static constexpr auto get_name(){return ((evalname+...+X::name)+my_static_string(")"));}
};


struct Domain{
  inline static constexpr auto name=my_static_string("domain");

  inline static constexpr auto evalname=name+my_static_string("(");
  template<class... X>
  inline static constexpr auto get_name(){return ((evalname+...+X::name)+my_static_string(")"));
  }
};
struct Span{
  inline static constexpr auto name=my_static_string("span(");
  template<class... X>
  inline static constexpr auto get_name(){return ((name+...+X::name)+my_static_string(")"));}
};
struct Map{
  inline static constexpr auto name=my_static_string("map(");
  template<class... X>
  inline static constexpr auto get_name(){return ((name+...+X::name)+my_static_string(")"));}
};
struct Size{
  inline static constexpr auto name=my_static_string("size(");
  template<class... X>
  inline static constexpr auto get_name(){return ((name+...+X::name)+my_static_string(")"));}
};


template<class ...Xs> struct span{
  inline static constexpr auto name=my_static_string("span");
  static   constexpr auto my_name(){return name;}

  span(){}
  static constexpr auto size(){return sizeof... (Xs);}
};


template<class T>
constexpr bool is_sorted(){return true;}

template<class T0,class T1,class...Ts>
constexpr bool is_sorted()
{
  return (T0::name<T1::name)&&is_sorted<T1,Ts...>();
}


template<class... X,class... Y,
          typename =std::enable_if_t<is_sorted<X...,Y...>()>>
auto operator&&(span<X...>,span<Y...>){
  return  span<X...,Y...>{};
}

auto operator*(span<>,span<>){return span<>{};}


template <class... Xs
          //,typename =std::enable_if_t<is_sorted<Xs...>()>
          >
auto operator*(span<>,span<Xs...>){return span<Xs...>{};}

template <class X,class... Xs
          // ,typename =std::enable_if_t<is_sorted<X,Xs...>()>
          >
auto operator*(span<X,Xs...>,span<>){return span<X,Xs...>{};}


template<class X,class Y, class...Xs, class...Ys,
          typename =std::enable_if_t<is_sorted<X,Xs...>()&&is_sorted<Y,Ys...>()>>
auto operator*(span<X,Xs...>,span<Y,Ys...>){
  if constexpr (std::is_same_v<X,Y >)
    return span<Y>{}&&(span<Xs...>{}*span<Ys...>{});
  else if constexpr (X::name<  Y::name)
    return span<X>{}&&(span<Xs...>{}*span<Y,Ys...>{});
  else
    return span<Y>{}&&(span<X,Xs...>{}*span<Ys...>{});
}



struct D{};



template<class F,class... Args>
struct Op
{
  inline static constexpr auto is_identifier=true;
  inline static constexpr auto is_operator=true;
  static constexpr auto name=F::template get_name<Args...>();
  static constexpr auto get_name(){return name;}

  __host__ __device__ Op(F,Args...){}
  __host__ __device__ Op(){}
};

template<class F>
struct Op<F>
{
  inline static constexpr auto is_identifier=true;
  inline static constexpr auto is_operator=true;
  static constexpr auto name=F::name;
  static constexpr auto get_name(){return name;}

  __host__ __device__ Op(F){}
  __host__ __device__ Op(){}
};


template <class Max>
auto index(Max n)
{
  if constexpr (std::is_integral_v<std::decay_t<Max>>)
  {
    using Int=std::decay_t<Max>;
    vector_device<Int> out(n);
    for (Int i=0; i<n; ++i)
      out[i]=i;
    return out;
  }
  else
  {
    return Op<Index,Max>{};
  }
}


template<class Min,class Max>
auto index(Min,Max)
{
  return Op<Index,Min,Max>{};
}


template<class X, class Y, typename=std::enable_if_t<X::is_identifier&&Y::is_identifier>>
constexpr auto operator+ (X,Y)
{
  if constexpr (X::name<Y::name)
    return Op<S,X,Y>{};
  else
    return Op<S,Y,X>{};
}

template<class X, typename=std::enable_if_t<X::is_identifier>>
constexpr auto operator+ (X,X)
{
  return Op<P,N<2>,X>{};
}







template<class X, class Y>
constexpr auto operator* (X,Y)->std::enable_if_t<X::is_identifier&&Y::is_identifier,
                                                   std::conditional_t<(X::name < Y::name),Op<P,X,Y>,Op<P,Y,X>>
                                                   >
{
  if constexpr (X::name < Y::name)
    return Op<P,X,Y>{};
  else
    return Op<P,Y,X>{};
}




template<class Id,class G>
class  F
{
public:
  typedef   Id myId;

  inline static constexpr auto name=Id::name+my_static_string("=")+G::name;
  auto &operator[](Id)const {return *this;}
  static   constexpr auto my_name(){return name;}

  constexpr F(Id ,G&& ){}
  constexpr F(){}
};

template<class Id,class G>
auto variables(F<Id,G>) {return Variables<Id>{};}


template<class Eval,class S, class... X>
auto arguments( Op<Eval,Op<S,X...>>)
{
  return (arguments(Op<Eval,X>{})+...);
}

template<class Id,class G>
constexpr auto arguments( Op<Eval,F<Id,G>>)
{
  return arguments(Op<Eval,G>{});
}

template<class Eval, class Id>
auto arguments(Op<Eval,Id>)
{
  return Arguments<Id>{};
}







struct serial_cpu;
struct parallel_cpu;
struct gpu;
struct global_gpu;
struct device_gpu;

//template< class Computer,class Operation,class...Xs>
//auto execute( Computer c,Operation, Xs...x);










#endif // CUDA_TEST_FUNCTIONS_H
