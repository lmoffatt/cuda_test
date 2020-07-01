#include <iostream>

#include "cuda_test_functions.h"
#include "cuda_test_cpu.h"
#include "cuda_test_gpu.h"

struct num
{
  inline static  constexpr auto is_identifier=true;
  inline static   constexpr auto name=my_static_string("num");
  static   constexpr auto my_name(){return name;}

};
struct t
{
  inline static  constexpr auto is_identifier=true;
   inline static   constexpr auto name=my_static_string("t");
   static   constexpr auto my_name(){return name;}
};


struct foo
{
  inline static  constexpr auto is_identifier=true;
  inline static constexpr   auto name=my_static_string("foo");
  static   constexpr auto my_name(){return name;}

};

struct bar
{
  inline static constexpr auto is_identifier=true;
  static   constexpr auto name=my_static_string("bar");
  static   constexpr auto my_name(){return name;}

};
template<class Id> struct delta_i{
  inline static constexpr auto is_identifier=true;
  inline static constexpr auto _name=my_static_string("delta_i(");
  inline static constexpr auto name=_name+Id::name+my_static_string(")");
  static   constexpr auto my_name(){return name;}
};

template<class Id> struct max_i{
  inline static constexpr auto is_identifier=true;
  inline static constexpr auto _name=my_static_string("max_i(");
  inline static constexpr auto name=_name+Id::name+my_static_string(")");
  static   constexpr auto my_name(){return name;}
};
template<class Id> struct myfun{
  inline static constexpr auto is_identifier=true;
  inline static constexpr auto myfun_name=my_static_string("myfun(");
  inline static constexpr auto name=myfun_name+Id::name+my_static_string(")");
  static   constexpr auto my_name(){return name;}
};

//template __global__ void executer2<global_gpu>();




int main()
{
  constexpr auto d=mapu(x_i(foo{},2ul),x_i(bar{},1),x_i(max_i<num>{},100),x_i(delta_i<t>{},0.1));
  auto q=quimulun{
      F(t{},index(max_i<num>{})*delta_i<t>{})
          ,
      F(myfun<foo>{},foo{}*t{}),
      F(myfun<bar>{},myfun<foo>{}+bar{})};


 // auto ar=arguments(Op(Eval{},q[myfun<bar>{}]));
   std::cout << "Hello World!" << std::endl;
  std::cout<<F(foo{},foo{}*bar{}).name.str()<<"\n";
  std::cout<<F(foo{},bar{}*foo{}).name.str()<<"\n";
  std::cout<<q.name.str()<<"\n";
 // executer2<global_gpu><<<1,1>>>();
  //exe3<<<1,100>>>();

  auto e=execute(serial_cpu{},Op(Eval{},q),d);
  std::cout<<e;
  //  std::cout<<out_type::size()<<"\n";

  auto g=execute(gpu{},Op(Eval{},q),d);
  cudaDeviceSynchronize();
  //using out_type=typename decltype (execute(gpu{},Op(Eval{},q),d))::hah;
  std::cout<<g;



  return 0;
}
