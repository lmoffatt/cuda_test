#ifndef CUDA_TEST_COMPILE_H
#define CUDA_TEST_COMPILE_H



#include "cuda_test_functions.h"
#include "cuda_test_transporter.h"


template< class Id,class Id2,class G,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Id>,Op<Assign,Id2,G>)
{
  if constexpr (std::is_same_v<Id,Id2>)
       return Op<Eval,Id>{};
    else
      return Nothing{};

}

template< class Id,class ...Ops,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Id>, non_sequential_block<Ops...>)
{
  return (Nothing{}+...+compile(device_gpu{},Op<Eval,Id>{},Ops{}));

}
template< class Id,class ...Ops,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Id>, sequential_block<Ops...>)
{
  return (Nothing{}+...+compile(device_gpu{},Op<Eval,Id>{},Ops{}));

}




template< class Id,class X,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Id>, Data<X>)
{
  if constexpr (std::is_same_v<Nothing,decltype (std::declval<X>()[Id{}]())>)
    return Nothing{};
  else return Op<Eval,Id>{};

}



template< class Id,class... Ops,class...Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Id>,sequential_block<Ops...> ops, Data<Xs...>x)
{
  //printf("execute( device_gpu,Op<Eval,Id>, Xs&&...x) \n ");
  //  using test=typename Id::Id;
  return ((Nothing{}+...+compile(device_gpu{},Op<Eval,Id>{},Ops{}))+...+compile(device_gpu{},Op<Eval,Id>{},Data<Xs>{}));
}

template< class Id,class Id2,class G,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain,Id>,Op<Assign,Id2,Map,span<>,G>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  if constexpr (std::is_same_v<Id,Id2>)
    return compile( device_gpu{},Op<Domain,G>{}, ops,x)*Id{};
  else
    return Nothing{};

}



template< class Id,class Id2,class Idx,class G,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain,Id>,Op<Assign,Id2,Map,span<Idx>,G>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  if constexpr (std::is_same_v<Id,Id2>)
    return span<Idx>{}*compile( device_gpu{},Op<Domain,G>{}, ops,x)*Id{};
  else
    return Nothing{};

}

template< class Id,class ...Ops,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain,Id>, non_sequential_block<Ops...>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  return (Nothing{}+...+compile(device_gpu{},Op<Domain,Id>{},Ops{},ops,x));

}



template< class Id,class ...Ops,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain,Id>, sequential_block<Ops...>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  return (Nothing{}+...+compile(device_gpu{},Op<Domain,Id>{},Ops{},ops,x));

}




template< class Id,class... xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain,Id>, Data<mapu<xs...>>)
{
  if constexpr (std::is_same_v<Nothing,decltype (std::declval<mapu<xs...>>()[Id{}]())>)
    return Nothing{};
  else
    return compile(device_gpu{},Op<Domain>{},Data<std::decay_t<decltype(std::declval<mapu<xs...>>()[Id{}]())>>{});
}



template< class Id,class... Ops,class...Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain,Id>,sequential_block<Ops...> ops, Data<Xs...>x)
{
  //printf("execute( device_gpu,Op<Eval,Id>, Xs&&...x) \n ");
  //  using test=typename Id::Id;
  return ((Nothing{}+...+compile(device_gpu{},Op<Domain,Id>{},Ops{},ops,x))+...+compile(device_gpu{},Op<Domain,Id>{},Data<Xs>{}));
}



template< class... X,class...Ops>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Op<P>>, Ops... x)
{
  return Op<P,Ops...>{};
}

template< class... X,class...Ops>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Op<S>>, Ops... x)
{
  return Op<S,Ops...>{};
}


template< class ...Idx, class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Map,Position<Idx...>>,Position<Idx...> ,T&& x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>>,Position<Idx...> ,T&& x) \n ");

  return std::forward<T>(x);
}

template<class ...Idx, class Idx0,class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Map,Position<Idx...>>,const Position<Idx...>& pos,const vector_field<T,Idx0>& x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>>,const Position<Idx...>& pos,const vector_field<T,Idx0>& x) \n ");
  return x(pos);
}


template< class Id,class ...Idx,class...Xs>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x)
{
  // printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x) \n ");
  return compile(device_gpu{},Op<Eval,Map,Position<Idx...>>{},pos,execute(device_gpu{},Op<Eval,Id>{},std::forward<Xs>(x)...));
}

template< class T>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain>, Data< T>)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x) \n ");
  return span<>{};
}
template< class T, class Idx>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Domain>, Data<vector_field<T,Idx>>)
{
  return span<Idx>{}*decltype(execute( device_gpu{},Op<Domain>{}, std::declval<const T&>())){};
}

template< class T, class Idx>
auto compile( device_gpu,Op<Eval,Size,span<Idx>>, const vector_field<T,Idx>& v)
{
  return v.size();
}
template< class T, class Idx, class Idx2>
__device__ __host__ auto compile( device_gpu,Op<Eval,Size,span<Idx>>, const vector_field<T,Idx2>& v)
{
  return Nothing{};
}

template< class T, class Idx>
__device__ __host__ auto compile( device_gpu,Op<Eval,Size,span<Idx>>, const T& )
{
  return Nothing{};
}

template< class... xs, class Idx>
__device__ __host__ auto compile( device_gpu,Op<Eval,Size,span<Idx>>, const mapu<xs...>& v)
{
  auto nothing=First_Result(Nothing{});
  return (First_Result<std::decay_t<decltype (compile(device_gpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))>>
          (compile(device_gpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))+...+nothing)();

}


template< class Idx,class X0,class X1,class...Xs>
__device__ __host__ auto compile( device_gpu,Op<Eval,Size,span<Idx>>, X0&& x0, X1&& x1, Xs&&...x)
{
  auto res0= First_Result(compile(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<X0>(x0)));
  auto res1=First_Result(compile(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<X1>(x1)));
  auto res= ((res0+res1)+...+First_Result(compile(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<Xs>(x))));
  return res();
}









template< class... X,class...Xs, class oP>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Op<oP,X...>>, Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Op<oP,X...>>, Xs&&...x)\n  ");

  return compile(device_gpu{},Op<Eval,Op<oP>>{},compile(device_gpu{},Op<Eval,X>{}, std::forward<Xs>(x)...)...);
}


template< class... X,class... Idx,class...Xs, class oP>
__device__ __host__ decltype(auto) compile( device_gpu,Op<Eval,Map,Position<Idx...>,Op<oP,X...>>, Position<Idx...> pos,Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Op<oP,X...>>, Position<Idx...> pos,Xs&&...x)\n  ");

  return compile(device_gpu{},Op<Eval,Op<oP>>{},compile(device_gpu{},Op<Eval,Map, Position<Idx...>,X>{},pos, std::forward<Xs>(x)...)...);
}
template< class oP,class... X,class...Xs>
__host__ __device__
    decltype(auto) compile( device_gpu,Op<Domain,Op<oP,X...>>, Xs...)
{
  //printf("execute( device_gpu,Op<Domain,Op<oP,X...>>, Xs&&...x)\n  ");
  if constexpr (std::is_same_v<Index,oP >)
    return span<self_span>{}*(compile(device_gpu{},Op<Domain,X>{}, Xs{}...)*...);
  else
     return (compile(device_gpu{},Op<Domain,X>{}, Xs{}...)*...);
}


template< class Max,class ...Ops,class...Xs>
__host__ __device__ auto compile( device_gpu,Op<Eval,Op<Index,Max>>,sequential_block<Ops...> ops, Data<Xs...>x)
{
  // using test=typename decltype (index(execute(device_gpu{},Op<Eval,Max>{}, std::forward<Xs>(x)...)))::te;


  return index(compile(device_gpu{},Op<Eval,Max>{},ops, x));
}

template<class G, class... Ops,class... Xs>
__host__ __device__ auto compile(device_gpu,Op<Eval,Map,span<>,G>,sequential_block<Ops...> ops,Data<Xs...> x)
{
  return compile(device_gpu{},Op(Eval{},G{}),ops, x);
}



template<class G,class Idx, class element_type,class... Xs>
__global__ void compile(int number,global_gpu,Op<Eval,Map,Position<Idx>,G>,element_type* v,  Xs...x)
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
auto compile(gpu,Op<Eval,Map,span<Idx>,G>,Xs&&...x)
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





template<class Id,class G, class... Ops,class... Xs>
auto compile(device_gpu,Op<Eval,F<Id,G>>,sequential_block<Ops...> ops,Data<Xs...> x)
{
  auto non_find_args=arguments(Op<Eval,G>{})-variables(ops)-variables(x);
  //using test=typename decltype (execute(serial_cpu{},Op<Eval,G>{}, std::forward<Xs>(x)...))::te;
  if constexpr (non_find_args.size==0)
  {
    auto mydomain=compile(device_gpu{},Op<Domain,G>{},ops,x)*Id{};
    return non_sequential_block(Op(Assign{},Id{},Map{},mydomain,G{}));
  }
  else
    return Error{Not_found{F<Id,G>{},non_find_args}};
}









template< class...Fs,class ...Ids,class... Ops,class...Xs>
constexpr auto compile( device_gpu,Op<Eval,quimulun<Fs...>,Variables<Ids...>>, sequential_block<Ops...> known_variables,Data<Xs...> x)
{
  auto new_variables= (compile(device_gpu{},Op(Eval{},quimulun<Fs...>{}[Ids{}]),known_variables,x)&&...);

  if constexpr (is_Error_v<decltype(new_variables)>)
    return new_variables;
  else
  {
    auto unknowns_vars=Variables<Ids...>{}-variables(new_variables);
    auto all_variables=std::move(known_variables)&&std::move(new_variables);
    if constexpr (unknowns_vars.size==0)
      return all_variables;
    else
      return compile(device_gpu{},Op(Eval{},quimulun<Fs...>{},unknowns_vars),std::move(all_variables),x);
  }

}








template< class quim,class...Xs>
constexpr auto compile( gpu,Op<Eval,quim>, Data<Xs...>)
{

  constexpr auto v=variables(quim{});
  // using test=typename decltype (execute(device_gpu{},Op(Eval{},quim{},v), mapu<>{}, std::forward<Xs>(x)...))::err;
  return compile(device_gpu{},Op(Eval{},quim{},v),sequential_block<>{}, Data<Xs...>{});
}










#endif // CUDA_TEST_COMPILE_H
