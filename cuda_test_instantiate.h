#ifndef CUDA_TEST_INSTANTIATE_H
#define CUDA_TEST_INSTANTIATE_H



#include "cuda_test_functions.h"
#include "cuda_test_transporter.h"


template<class F, class D>
class Monad
{
private:
  F f_;
  D d_;
public:
  auto& f()&{return f_;}
  auto& f()const &{return f_;}
  auto&& f()&& {return std::move(f_);}
  auto& d()&{return d_;}
  auto& d()const &{return d_;}
  auto&& d()&& {return std::move(d_);}

  Monad(F&& f, D&& d): f_{std::move(f)},d_{std::move(d)}{}
  Monad(F const& f, D const& d): f_{f},d_{d}{}
  Monad(){}
  template<class... xs2>
  friend decltype(auto) operator&&(Monad&& out, Error<xs2...>){return std::move(out);}

  template<class...xs2>
  friend decltype(auto) operator&&(Error<xs2...>,Monad&& out){return std::move(out);}



  template<class F2, class D2>
  friend auto operator&&(Monad&& one, Monad<F2,D2>&& two){
    auto outf=std::move(one).f()&&std::move(two).f();
    auto outd=std::move(one).d()&&std::move(two).d();
    return Monad<decltype (outf),decltype (outd)>{};
  }


};
template<class... >
class Data_new;

template<class Id, typename T>
class Data_new< x_i<Id,T>>
{

public:

  inline static constexpr auto x_iname=my_static_string("x_i(");
  inline static constexpr auto name=x_iname+Id::name+my_static_string(")");
  using myId=Id;
      constexpr auto operator[](Id)const {return *this;}

};

template<class Id>
class Data_new< ind<Id>>
{

public:

  inline static constexpr auto x_iname=my_static_string("x_i(");
  inline static constexpr auto name=x_iname+Id::name+my_static_string(")");
  using myId=Id;
  constexpr auto operator[](Id)const {return *this;}

};

template<class Id, class index_type>
class Data_new< ind<index_type,Id>>
{

public:

  inline static constexpr auto x_iname=my_static_string("x_i(");
  inline static constexpr auto name=x_iname+Id::name+my_static_string(")");
  using myId=Id;
  constexpr auto operator[](Id)const {return *this;}

};


template<class... xs>
struct Data_new:
    public Data_new<xs>...{
  using Data_new<xs>::operator[]...;

  template<class...xxs,typename=std::enable_if<((sizeof...(xs)>0)&&... && std::is_same_v<xxs,xs>) > >
  Data_new(Data_new<xxs>...){}
  Data_new()=default;

    inline static constexpr auto data_name=my_static_string("data");
    inline static constexpr auto name=(data_name+...+xs::name);

    constexpr inline static auto size=sizeof... (xs);


    template<class Id>
    __host__ __device__
        Nothing operator[](Id)const {return Nothing{};}




    template<class ... xs2>
    __host__ __device__
        friend auto operator&&(Data_new, Data_new<xs2...> )
    {
      return Data_new<xs...,xs2...>{};
    }
  };





  template <class...xs>
  constexpr auto variables_new(Data_new<xs...>)
  {
    //  using test=typename Cs<xs...,
    //                           decltype ((variables(std::declval<xs>())&&...&&Variables<>{})),
    //                           decltype (variables(std::declval<xs>()))...>::var;
    return decltype ((variables_new(std::declval<xs>())&&...&&Variables<>{})){};
  }


  template<class F,class D>
  constexpr auto variables_new(Monad<F,D>) {return variables_new(D{});}

  template<class > struct span_new;
  template<> struct span_new<ind<>>
  {
    inline static constexpr auto span_name=my_static_string("span(");
    inline static constexpr auto name=span_name+my_static_string(")");
    static   constexpr auto my_name(){return name;}

    span_new(ind<>){}
    span_new(){}


    template<class X2>
    friend auto operator* (span_new,span_new<X2>)
    {
      return span_new<X2>{};
    }

    template<class X2>
    friend auto operator* (span_new<X2>,span_new)
    {
      return span_new<X2>{};
    }

    friend auto operator* (span_new,span_new )
    {
      return span_new{};
    }
    static constexpr auto size(){return 0;}
  };

  template<class index_type> struct span_new
  {
    inline static constexpr auto span_name=my_static_string("span(");
    inline static constexpr auto name=(span_name+index_type::name)+my_static_string(")");
    static   constexpr auto my_name(){return name;}

    span_new(index_type){}
    span_new(){}


    template<class ...X2>
    friend auto operator* (span_new,span_new<ind_prod<X2...>>)
    {
      using Xout=decltype (std::declval<ind_prod<Xs...>>()*std::declval<ind_prod<X2...>>());
      return span_new<Xout>{};
    }

    friend auto operator* (span_new ,span_new<ind<>>)
    {
      return span_new{};
    }

    friend auto operator* (span_new<ind<>>,span_new )
    {
      return span_new{};
    }
    static constexpr auto size(){return sizeof... (Xs);}
  };

  template<class ...Xs> struct span_new<ind_prod<Xs...>>
  {
    inline static constexpr auto span_name=my_static_string("span(");
    inline static constexpr auto name=(span_name+ind_prod<Xs...>::name)+my_static_string(")");
    static   constexpr auto my_name(){return name;}

    span_new(ind_prod<Xs...>){}
    span_new(){}


    template<class ...X2>
    friend auto operator* (span_new,span_new<ind_prod<X2...>>)
    {
      using Xout=decltype (std::declval<ind_prod<Xs...>>()*std::declval<ind_prod<X2...>>());
      return span_new<Xout>{};
    }

    static constexpr auto size(){return sizeof... (Xs);}
  };

  template <class Id, class T, class... Ids>
  auto build_new(Address<Id>,Data_new<T>, span_new<Ids...>)
  {

      return Op<Build_new,Address<Id>,T,span_new<Ids...>>{};
  }
  
  template <class Id, class T, class... Ids>
  auto map_new(Address<Id>,Data_new<T>, span_new<Ids...>)
  {
    
    return Op<Build_new,Address<Id>,T,span_new<Ids...>>{};
  }


  template<class Id, typename T>
  constexpr auto domain_new(const x_i<Id,T>&)
  {
    return ind<>{};
  }

  template<class Id>
  constexpr auto& domain_new(const ind<Id>& x)
  {
    return x;
  }



  constexpr auto domain_new(Nothing)
  {
    return Nothing{};
  }


  template<class Id, class Value_type,class indice_type>
  auto& domain_new(const x_i<Id,vector_by_index<Value_type,indice_type>>& x)
  {
    return x().index();
  }








  template <class T>
  auto domain_new(Data_new<T>){
    using sp=std::decay_t<decltype (domain_new(std::declval<T>()))>;
    return span_new<sp>{};
  }

  template<class X, class... xs>
  decltype(auto) domain_new(X,Data_new<xs...> x)
  {
    return domain_new(x[X{}]);
  }
  template<class O, class ...X, class ...xs>
  auto domain_new(Op<O,X...>,Data_new<xs...> x)
  {
    return (domain_new(x[X{}])*...*span_new<ind<>>{});
  }






template< class Id,class Id2,class G,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Id>,Op<Assign,Id2,G>)
{
  if constexpr (std::is_same_v<Id,Id2>)
    return Op<Eval,Id>{};
  else
    return Nothing{};

}

template< class Id,class ...Ops,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Id>, non_sequential_block<Ops...>)
{
  return (Nothing{}+...+instantiate(device_gpu{},Op<Eval,Id>{},Ops{}));

}



template< class Id,class ...Ops,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Id>, sequential_block<Ops...>)
{
  return (Nothing{}+...+instantiate(device_gpu{},Op<Eval,Id>{},Ops{}));

}




template< class Id,class X,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Id>, Data<X>)
{
  if constexpr (std::is_same_v<Nothing,decltype (std::declval<X>()[Id{}]())>)
    return Nothing{};
  else return Op<Eval,Id>{};

}



template< class Id,class... Ops,class...Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Id>,sequential_block<Ops...> ops, Data<Xs...>x)
{
  //printf("execute( device_gpu,Op<Eval,Id>, Xs&&...x) \n ");
  //  using test=typename Id::Id;
  return ((Nothing{}+...+instantiate(device_gpu{},Op<Eval,Id>{},Ops{}))+...+instantiate(device_gpu{},Op<Eval,Id>{},Data<Xs>{}));
}



template< class Id,class Id2,class G,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain,Id>,Op<Assign,Id2,Map,span<>,G>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  if constexpr (std::is_same_v<Id,Id2>)
    return instantiate( device_gpu{},Op<Domain,G>{}, ops,x)*Id{};
  else
    return Nothing{};

}



template< class Id,class Id2,class Idx,class G,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain,Id>,Op<Assign,Id2,Map,span<Idx>,G>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  if constexpr (std::is_same_v<Id,Id2>)
    return span<Idx>{}*instantiate( device_gpu{},Op<Domain,G>{}, ops,x)*Id{};
  else
    return Nothing{};

}

template< class Id,class ...Ops,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain,Id>, non_sequential_block<Ops...>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  return (Nothing{}+...+instantiate(device_gpu{},Op<Domain,Id>{},Ops{},ops,x));

}



template< class Id,class ...Ops,class ...OOps, class... Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain,Id>, sequential_block<Ops...>,sequential_block<OOps...> ops,Data<Xs...> x)
{
  return (Nothing{}+...+instantiate(device_gpu{},Op<Domain,Id>{},Ops{},ops,x));

}




template< class Id,class... xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain,Id>, Data<mapu<xs...>>)
{
  if constexpr (std::is_same_v<Nothing,decltype (std::declval<mapu<xs...>>()[Id{}]())>)
    return Nothing{};
  else
    return instantiate(device_gpu{},Op<Domain>{},Data<std::decay_t<decltype(std::declval<mapu<xs...>>()[Id{}]())>>{});
}



template< class Id,class... Ops,class...Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain,Id>,sequential_block<Ops...> ops, Data<Xs...>x)
{
  //printf("execute( device_gpu,Op<Eval,Id>, Xs&&...x) \n ");
  //  using test=typename Id::Id;
  return ((Nothing{}+...+instantiate(device_gpu{},Op<Domain,Id>{},Ops{},ops,x))+...+instantiate(device_gpu{},Op<Domain,Id>{},Data<Xs>{}));
}



template< class... X,class...Ops>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Op<P>>, Ops... x)
{
  return Op<P,Ops...>{};
}

template< class... X,class...Ops>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Op<S>>, Ops... x)
{
  return Op<S,Ops...>{};
}


template< class ...Idx, class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Map,Position<Idx...>>,Position<Idx...> ,T&& x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>>,Position<Idx...> ,T&& x) \n ");

  return std::forward<T>(x);
}

template<class ...Idx, class Idx0,class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Map,Position<Idx...>>,const Position<Idx...>& pos,const vector_field<T,Idx0>& x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>>,const Position<Idx...>& pos,const vector_field<T,Idx0>& x) \n ");
  return x(pos);
}


template< class Id,class ...Idx,class...Xs>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x)
{
  // printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x) \n ");
  return instantiate(device_gpu{},Op<Eval,Map,Position<Idx...>>{},pos,execute(device_gpu{},Op<Eval,Id>{},std::forward<Xs>(x)...));
}

template< class T>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain>, Data< T>)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x) \n ");
  return span<>{};
}
template< class T, class Idx>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Domain>, Data<vector_field<T,Idx>>)
{
  return span<Idx>{}*decltype(execute( device_gpu{},Op<Domain>{}, std::declval<const T&>())){};
}

template< class T, class Idx>
auto instantiate( device_gpu,Op<Eval,Size,span<Idx>>, const vector_field<T,Idx>& v)
{
  return v.size();
}
template< class T, class Idx, class Idx2>
__device__ __host__ auto instantiate( device_gpu,Op<Eval,Size,span<Idx>>, const vector_field<T,Idx2>& v)
{
  return Nothing{};
}

template< class T, class Idx>
__device__ __host__ auto instantiate( device_gpu,Op<Eval,Size,span<Idx>>, const T& )
{
  return Nothing{};
}

template< class... xs, class Idx>
__device__ __host__ auto instantiate( device_gpu,Op<Eval,Size,span<Idx>>, const mapu<xs...>& v)
{
  auto nothing=First_Result(Nothing{});
  return (First_Result<std::decay_t<decltype (instantiate(device_gpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))>>
          (instantiate(device_gpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))+...+nothing)();

}


template< class Idx,class X0,class X1,class...Xs>
__device__ __host__ auto instantiate( device_gpu,Op<Eval,Size,span<Idx>>, X0&& x0, X1&& x1, Xs&&...x)
{
  auto res0= First_Result(instantiate(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<X0>(x0)));
  auto res1=First_Result(instantiate(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<X1>(x1)));
  auto res= ((res0+res1)+...+First_Result(instantiate(device_gpu{},Op<Eval,Size,span<Idx>>{},std::forward<Xs>(x))));
  return res();
}









template< class... X,class...Xs, class oP>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Op<oP,X...>>, Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Op<oP,X...>>, Xs&&...x)\n  ");

  return instantiate(device_gpu{},Op<Eval,Op<oP>>{},instantiate(device_gpu{},Op<Eval,X>{}, std::forward<Xs>(x)...)...);
}


template< class... X,class... Idx,class...Xs, class oP>
__device__ __host__ decltype(auto) instantiate( device_gpu,Op<Eval,Map,Position<Idx...>,Op<oP,X...>>, Position<Idx...> pos,Xs&&...x)
{
  //printf("execute( device_gpu,Op<Eval,Map,Position<Idx...>,Op<oP,X...>>, Position<Idx...> pos,Xs&&...x)\n  ");

  return instantiate(device_gpu{},Op<Eval,Op<oP>>{},instantiate(device_gpu{},Op<Eval,Map, Position<Idx...>,X>{},pos, std::forward<Xs>(x)...)...);
}

template< class oP,class... X,class...Xs>
auto domain(Op<oP,X...>, Data_new<Xs...>)
{
  //printf("execute( device_gpu,Op<Domain,Op<oP,X...>>, Xs&&...x)\n  ");
  if constexpr (std::is_same_v<Index,oP >)
    return span<self_span>{}*(instantiate(device_gpu{},Op<Domain,X>{}, Xs{}...)*...);
  else
    return (instantiate(device_gpu{},Op<Domain,X>{}, Xs{}...)*...);
}


template< class Max,class ...Ops,class...Xs>
__host__ __device__ auto instantiate( device_gpu,Op<Eval,Op<Index,Max>>,sequential_block<Ops...> ops, Data<Xs...>x)
{
  // using test=typename decltype (index(execute(device_gpu{},Op<Eval,Max>{}, std::forward<Xs>(x)...)))::te;


  return index(instantiate(device_gpu{},Op<Eval,Max>{},ops, x));
}

template<class G, class... Ops,class... Xs>
__host__ __device__ auto instantiate(device_gpu,Op<Eval,Map,span<>,G>,sequential_block<Ops...> ops,Data<Xs...> x)
{
  return instantiate(device_gpu{},Op(Eval{},G{}),ops, x);
}



template<class G,class Idx, class element_type,class... Xs>
__global__ void instantiate(int number,global_gpu,Op<Eval,Map,Position<Idx>,G>,element_type* v,  Xs...x)
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
auto instantiate(gpu,Op<Eval,Map,span<Idx>,G>,Xs&&...x)
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



template<class Id,class G, class...xs>
auto instantiate(Op<Index_new,Id,Map,span_new<ind<>>,G>,Data_new<xs...> )
{


  return Monad(quimulun{
                   F(Id{},G{})
               },
               Data_new<
                   ind<Id>
                   >{} );
}
template<class Id,class X, class...xs>
auto instantiate(Op<Eval,Op<X>>,Data_new<xs...> x)
{
  return x[X{}]();

}

template<class Id,class O,class...X, class...xs>
auto instantiate(Op<Eval,Op<S>>,xs... x)
{
  return Op<Eval,std::decay_t<decltype ((x+....))>{};
}
template<class Id,class O,class...X, class...xs>
auto instantiate(Op<Eval,Op<O,X...>>,Data_new<xs...> x)
{
  return intantiate(Op<Eval,Op<O>>{},instantiate(Op<Eval,X>{},x)...);
}




template<class Id,class G, class...Idx,class...xs>
auto instantiate(Op<Assign,Id,Map,span_new<ind_prod<Idx...>>,G>,Data_new<xs...> )
{

  using  T=double;
  
  using Pos=Position<Idx...>; //wrong
  using index_type=ind_prod<Idx...>; 
  
  return Monad(quimulun{
                   F(Address<Id>{},build_new(Data_new<T>{},span_new<ind_prod<Idx...>>{})),
                   F(Element<Id,Pos>{},build_new(Adress<Id>,Data_new<T>{},span_new<ind_prod<Idx...>>{})),
                   F(Id{},build_new(Data_new<T>{},span_new<ind_prod<Idx...>>{})),
                   
               },
               Data_new<
                   x_i<Id,vector_by_index<T,index_type>>
                   >{} );
}



template<class Id,class G, class...xs>
auto instantiate(Op<Assign,Id,Map,span_new<ind<>>,G>,Data_new<xs...> )
{


  return Monad(quimulun{
                   F(Id{},G{})
               },
               Data_new<
                   ind<Id>
                   >{} );
}



template<class Id,class G,class...xs>
auto instantiate(Op<Eval,I_new<Id,G>>,Data_new<xs...> x)
{
  auto non_find_args=arguments(Op<Eval,G>{})-variables_new(x);
  //using test=typename decltype (execute(serial_cpu{},Op<Eval,G>{}, std::forward<Xs>(x)...))::te;
  if constexpr (non_find_args.size==0)
  {
    auto mydomain=domain_new(G{},Data_new<xs...>{});
    //using test=typename Cs<decltype (mydomain),G>::tetd;
    return instantiate(Op(Index_new{},Id{},Map{},mydomain,G{}),x);
  }
  else
    return Error{Not_found{F<Id,G>{},non_find_args}};
}



template<class Id,class G, class...xs>
auto instantiate(Op<Eval,F<Id,G>>,Data_new<xs...>  x)
{
  auto non_find_args=arguments(Op<Eval,G>{})-variables_new(x);
  //using test=typename decltype (execute(serial_cpu{},Op<Eval,G>{}, std::forward<Xs>(x)...))::te;
  if constexpr (non_find_args.size==0)
  {
    auto mydomain=domain_new(G{},Data_new<xs...>{});
    return instantiate(Op(Assign{},Id{},Map{},mydomain,G{}),Data_new<xs...>{});
  }
  else
    return Error{Not_found{F<Id,G>{},non_find_args}};
}









template< class...Fs,class ...Ids,class F, class... xs,class...Xs>
constexpr auto instantiate( Op<Eval,quimulun<Fs...>,Variables<Ids...>>,Monad<F,Data_new<xs...>> known_variables,Data_new<Xs...> x )
{
  auto new_variables= (instantiate(Op(Eval{},quimulun<Fs...>{}[Ids{}]),Data_new<xs...,Xs...>{})&&...);

  if constexpr (is_Error_v<decltype(new_variables)>)
    return new_variables;
  else
  {
    auto unknowns_vars=Variables<Ids...>{}-variables_new(new_variables);
    auto all_variables=std::move(known_variables)&&std::move(new_variables);
    if constexpr (unknowns_vars.size==0)
      return all_variables;
    else
      return instantiate(Op(Eval{},quimulun<Fs...>{},unknowns_vars),all_variables,x);
  }

}








template< class quim,class...Xs>
constexpr auto instantiate(Op<Eval,quim>, mapu<Xs...>)
{

  constexpr auto v=variables_new(quim{});
  // using test=typename decltype (execute(device_gpu{},Op(Eval{},quim{},v), mapu<>{}, std::forward<Xs>(x)...))::err;
  return instantiate(Op(Eval{},quim{},v),Monad<quimulun<>,Data_new<>>{}, Data_new<Xs...>{});
}













#endif // CUDA_TEST_INSTANTIATE_H
