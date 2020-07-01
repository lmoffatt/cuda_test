#ifndef CUDA_TEST_GPU_H
#define CUDA_TEST_GPU_H
#include "cuda_test_functions.h"
// CUDA runtime
#include "cuda_runtime.h"
#include "cuda_test_transporter.h"
// helper functions and utilities to work with CUDA
//#include <helper_cuda.h>
//#include <helper_functions.h>

struct global_gpu{
  inline static constexpr auto num_threads=100;


};
struct device_gpu{
 // __device__ __host__
  device_gpu()=default;
};
struct gpu{};

template< class Id,class...Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Id>, Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Id>, Xs&&...x) \n ");
  //  using test=typename Id::Id;
  return (Nothing{}+...+std::forward<Xs>(x)[Id{}]());
}

template< class... X,class...Xs>
__device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Op<P>>, Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Op<P>>, Xs&&...x) \n ");

  return (std::forward<Xs>(x)*...);
}

template< class... X,class...Xs>
__device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Op<S>>, Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Op<S>>, Xs&&...x) \n ");

  return (std::forward<Xs>(x)+...);
}


template< class ...Idx, class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
__device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Map,Position<Idx...>>,Position<Idx...> ,T&& x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>>,Position<Idx...> ,T&& x) \n ");

  return std::forward<T>(x);
}

template<class ...Idx, class Idx0,class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
__device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Map,Position<Idx...>>,const Position<Idx...>& pos,const vector_field<T,Idx0>& x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>>,const Position<Idx...>& pos,const vector_field<T,Idx0>& x) \n ");
  return x(pos);
}


template< class Id,class ...Idx,class...Xs>
__device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x)
{
 // printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x) \n ");
  return execute(device_gpu{},Op<Eval,Map,Position<Idx...>>{},pos,execute(device_gpu{},Op<Eval,Id>{},std::forward<Xs>(x)...));
}

template< class T>
__device__ __host__ decltype(auto) execute( device_gpu,Op<Domain>, const T&)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x) \n ");
  return span<>{};
}
template< class T, class Idx>
__device__ __host__ decltype(auto) execute( device_gpu,Op<Domain>, const vector_field<T,Idx>&)
{
  return span<Idx>{}*decltype(execute( device_gpu{},Op<Domain>{}, std::declval<const T&>())){};
}

template< class T, class Idx>
auto execute( device_gpu,Op<Eval,Size,span<Idx>>, const vector_field<T,Idx>& v)
{
    return v.size();
}
template< class T, class Idx, class Idx2>
 __device__ __host__ auto execute( device_gpu,Op<Eval,Size,span<Idx>>, const vector_field<T,Idx2>& v)
{
    return Nothing{};
}

template< class T, class Idx>
 __device__ __host__ auto execute( device_gpu,Op<Eval,Size,span<Idx>>, const T& )
{
  return Nothing{};
}

template< class... xs, class Idx>
 __device__ __host__ auto execute( device_gpu,Op<Eval,Size,span<Idx>>, const mapu<xs...>& v)
{
  auto nothing=First_Result(Nothing{});
  return (First_Result<std::decay_t<decltype (execute(device_gpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))>>
          (execute(device_gpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))+...+nothing)();

}


template< class Idx,class X0,class X1,class...Xs>
 __device__ __host__ auto execute( device_gpu,Op<Eval,Size,span<Idx>>, X0&& x0, X1&& x1, Xs&&...x)
{
    auto res0= First_Result(execute(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<X0>(x0)));
    auto res1=First_Result(execute(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<X1>(x1)));
    auto res= ((res0+res1)+...+First_Result(execute(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<Xs>(x))));
    return res();
}




template< class Id,class...Xs>
 __device__ __host__ decltype(auto) execute( device_gpu,Op<Domain,Id>, Xs&&...x)
{
  //printf(" execute( device_gpu,Op<Domain,Id>, Xs&&...x) \n  ");
  return execute(device_gpu{},Op<Domain>{},execute(device_gpu{},Op<Eval,Id>{},std::forward<Xs>(x)...));
}





template< class... X,class...Xs, class oP>
 __device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Op<oP,X...>>, Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Op<oP,X...>>, Xs&&...x)\n  ");

  return execute(device_gpu{},Op<Eval,Op<oP>>{},execute(device_gpu{},Op<Eval,X>{}, std::forward<Xs>(x)...)...);
}


template< class... X,class... Idx,class...Xs, class oP>
 __device__ __host__ decltype(auto) execute( device_gpu,Op<Eval,Map,Position<Idx...>,Op<oP,X...>>, Position<Idx...> pos,Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Op<oP,X...>>, Position<Idx...> pos,Xs&&...x)\n  ");

  return execute(device_gpu{},Op<Eval,Op<oP>>{},execute(device_gpu{},Op<Eval,Map, Position<Idx...>,X>{},pos, std::forward<Xs>(x)...)...);
}
template< class oP,class... X,class...Xs>
__host__ __device__
    decltype(auto) execute( device_gpu,Op<Domain,Op<oP,X...>>, Xs&&...x)
{
  //printf("execute( device_gpu,Op<Domain,Op<oP,X...>>, Xs&&...x)\n  ");
  return (execute(device_gpu{},Op<Domain,X>{}, std::forward<Xs>(x)...)*...);
}


template< class Max,class...Xs>
__host__ __device__ auto execute( device_gpu,Op<Eval,Op<Index,Max>>, Xs&&...x)
{
  // using test=typename decltype (index(execute(device_gpu{},Op<Eval,Max>{}, std::forward<Xs>(x)...)))::te;


  return index(execute(device_gpu{},Op<Eval,Max>{}, std::forward<Xs>(x)...));
}

template<class G, class... Xs>
__host__ __device__ auto execute(device_gpu,Op<Eval,Map,span<>,G>,Xs&&...x)
{
  return execute(device_gpu{},Op(Eval{},G{}), std::forward<Xs>(x)...);
}
template<class G, class... Xs>
__host__ __device__ auto execute(gpu,Op<Eval,Map,span<>,G>,Xs&&...x)
{
  return execute(device_gpu{},Op(Eval{},G{}), std::forward<Xs>(x)...);
}



template<class G,class Idx, class element_type,class... Xs>
__global__ void execute(int number,global_gpu,Op<Eval,Map,Position<Idx>,G>,element_type* v,  Xs...x)
{
  int id=threadIdx.x;
  Position<Idx> p(threadIdx.x);
  if (p()==0)
  {
    printf("\n\n-----aqui----\n\n");
    (my_print(x()),...);
    printf("\n\n-----aqui----\n\n\n");
  }
  // printf("p()=%4d n= %g %s\n",p(),nu,G::get_name().c_str());
  if (p()<number)
  {
  //  printf("id=%d \n",id);
 //  printf("\n\n-----aqui----\n\n\n");

//    (my_print(x()),...);
 //   printf("\n\n-----aqui----\n\n\n");
 //   printf("resc[%d] = %g \n", p(), nu);
    v[p()]=execute(device_gpu{},Op<Eval,Map,Position<Idx>,G>{},p,x()...);
   // printf("resc[%d] = %g \n", id, resx);
  }
}

//template __global__ void executer<global_gpu>(global_gpu);

template<class G,class Idx, class... Xs>
auto execute(gpu,Op<Eval,Map,span<Idx>,G>,Xs&&...x)
{
  int n=execute(device_gpu{},Op<Eval,Size,span<Idx>>{}, std::forward<Xs>(x)...);

  using element_type=std::decay_t<decltype (execute(device_gpu{},Op<Eval,Map,Position<Idx>,G>{},Position<Idx>{}, std::forward<Xs>(x)...))>;

  auto vector_out=vector_device<element_type>(n);
  element_type *out;
  auto size=n*sizeof (element_type);
  cudaMalloc(&out, size);
  std::cout<<"n es "<<n<<"\n";
  execute<<<1,global_gpu::num_threads>>>(n,global_gpu{},Op<Eval,Map,Position<Idx>,G>{},out, send_to_device(x)...);
  cudaMemcpy( &vector_out[0],out, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  {
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n",
             cudaGetErrorString(cudaerr));
  }

   return vector_field<element_type,Idx>(std::move(vector_out));
}


template<class G,class Idx, class... Xs>
auto execute(device_gpu,Op<Eval,Map,span<Idx>,G>,Xs&&...x)
{
  auto n=execute(device_gpu{},Op<Eval,Size,span<Idx>>{}, std::forward<Xs>(x)...);

  using element_type=std::decay_t<decltype (execute(device_gpu{},Op<Eval,Map,Position<Idx>,G>{},Position<Idx>{}, std::forward<Xs>(x)...))>;

  auto vector_out=vector_device<element_type>(n);

  for (Position<Idx> p{}; p()<n; ++p())
    vector_out[p()]=execute(device_gpu{},Op<Eval,Map,Position<Idx>,G>{},p, std::forward<Xs>(x)...);
  return vector_field<element_type,Idx>(std::move(vector_out));
}




template<class Id,class G, class... Xs>
auto execute(device_gpu,Op<Eval,F<Id,G>>,Xs&&...x)
{
  auto non_find_args=arguments(Op<Eval,G>{})-(variables(x)&&...);
  //using test=typename decltype (execute(serial_cpu{},Op<Eval,G>{}, std::forward<Xs>(x)...))::te;
  if constexpr (non_find_args.size==0)
  {
    auto mydomain=execute(device_gpu{},Op<Domain,G>{},std::forward<Xs>(x)...);
    return mapu(x_i(Id{},execute(gpu{},Op(Eval{},Map{},mydomain,G{}), std::forward<Xs>(x)...)));
  }
  else
    return Error{Not_found{F<Id,G>{},non_find_args}};
}









template< class...Fs,class ...Ids,class... xs,class...Xs>
auto execute( device_gpu,Op<Eval,quimulun<Fs...>,Variables<Ids...>>, mapu<xs...>&& known_variables,Xs&&...x)
{
  auto new_variables= (execute(device_gpu{},Op(Eval{},quimulun<Fs...>{}[Ids{}]),known_variables,std::forward<Xs>(x)...)&&...);

  if constexpr (is_Error_v<decltype(new_variables)>)
    return new_variables;
  else
  {
    auto unknowns_vars=Variables<Ids...>{}-variables(new_variables);
    auto all_variables=std::move(known_variables)&&std::move(new_variables);
    if constexpr (unknowns_vars.size==0)
      return all_variables;
    else
      return execute(device_gpu{},Op(Eval{},quimulun<Fs...>{},unknowns_vars),std::move(all_variables),std::forward<Xs>(x)...);
  }

}








template< class quim,class...Xs>
auto execute( gpu,Op<Eval,quim>, Xs&&...x)
{

  auto v=variables(quim{});
 // using test=typename decltype (execute(device_gpu{},Op(Eval{},quim{},v), mapu<>{}, std::forward<Xs>(x)...))::err;
  return execute(device_gpu{},Op(Eval{},quim{},v), mapu<>{}, std::forward<Xs>(x)...);
}










#endif // CUDA_TEST_GPU_H
