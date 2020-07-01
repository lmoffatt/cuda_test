#ifndef CUDA_TEST_CPU_H
#define CUDA_TEST_CPU_H


#include "cuda_test_functions.h"


struct serial_cpu{};

template< class Id,class...Xs,typename=std::enable_if_t<!is_any_of_these_template_classes<Op,quimulun,F>::template value<Id>> >
decltype(auto) execute( serial_cpu,Op<Eval,Id>, Xs&&...x)
{
//  using test=typename Id::Id;
  return (Nothing{}+...+std::forward<Xs>(x)[Id{}]());
}

template< class... X,class...Xs>
decltype(auto) execute( serial_cpu,Op<Eval,Op<P>>, Xs&&...x)
{
  return (std::forward<Xs>(x)*...);
}

template< class... X,class...Xs>
decltype(auto) execute( serial_cpu,Op<Eval,Op<S>>, Xs&&...x)
{
  return (std::forward<Xs>(x)+...);
}


template< class ...Idx, class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
decltype(auto) execute( serial_cpu,Op<Eval,Map,Position<Idx...>>,Position<Idx...> ,T&& x)
{
  return std::forward<T>(x);
}

template<class ...Idx, class Idx0,class T, typename=std::enable_if_t<!is_this_template_class_v<vector_field,T>>>
decltype(auto) execute( serial_cpu,Op<Eval,Map,Position<Idx...>>,const Position<Idx...>& pos,const vector_field<T,Idx0>& x)
{
  return x(pos);
}


template< class Id,class ...Idx,class...Xs>
decltype(auto) execute( serial_cpu,Op<Eval,Map,Position<Idx...>,Id>,Position<Idx...> pos, Xs&&...x)
{
  return execute(serial_cpu{},Op<Eval,Map,Position<Idx...>>{},pos,execute(serial_cpu{},Op<Eval,Id>{},std::forward<Xs>(x)...));
}

template< class T>
decltype(auto) execute( serial_cpu,Op<Domain>, const T&)
{
  return span<>{};
}
template< class T, class Idx>
decltype(auto) execute( serial_cpu,Op<Domain>, const vector_field<T,Idx>&)
{
  return span<Idx>{}*decltype(execute( serial_cpu{},Op<Domain>{}, std::declval<const T&>())){};
}

template< class T, class Idx, class Idx2>
auto execute( serial_cpu,Op<Eval,Size,span<Idx>>, const vector_field<T,Idx2>& v)
{
  if constexpr (std::is_same_v<Idx,Idx2 >)
     return v.size();
  else
    return Nothing{};
}
template< class T, class Idx>
auto execute( serial_cpu,Op<Eval,Size,span<Idx>>, const T& )
{
  return Nothing{};
}

template< class... xs, class Idx>
auto execute( serial_cpu,Op<Eval,Size,span<Idx>>, const mapu<xs...>& v)
{
    auto nothing=First_Result(Nothing{});
    return (First_Result<std::decay_t<decltype (execute(serial_cpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))>>
            (execute(serial_cpu{},Op<Eval,Size,span<Idx>>{},v[typename xs::myId{}]()))+...+nothing)();
}


template< class Idx,class X0,class X1,class...Xs>
auto execute( serial_cpu,Op<Eval,Size,span<Idx>>, X0&& x0, X1&& x1, Xs&&...x)
{
 auto res0= First_Result(execute(serial_cpu{},Op<Eval,Size,span<Idx>>{},std::forward<X0>(x0)));
auto res1=First_Result(execute(serial_cpu{},Op<Eval,Size,span<Idx>>{},std::forward<X1>(x1)));
auto res= ((res0+res1)+...+First_Result(execute(serial_cpu{},Op<Eval,Size,span<Idx>>{},std::forward<Xs>(x))));
  return res();
}




template< class Id,class...Xs>
decltype(auto) execute( serial_cpu,Op<Domain,Id>, Xs&&...x)
{
  return execute(serial_cpu{},Op<Domain>{},execute(serial_cpu{},Op<Eval,Id>{},std::forward<Xs>(x)...));
}





template< class... X,class...Xs, class oP>
decltype(auto) execute( serial_cpu,Op<Eval,Op<oP,X...>>, Xs&&...x)
{
  return execute(serial_cpu{},Op<Eval,Op<oP>>{},execute(serial_cpu{},Op<Eval,X>{}, std::forward<Xs>(x)...)...);
}


template< class... X,class... Idx,class...Xs, class oP>
decltype(auto) execute( serial_cpu,Op<Eval,Map,Position<Idx...>,Op<oP,X...>>, Position<Idx...> pos,Xs&&...x)
{
  return execute(serial_cpu{},Op<Eval,Op<oP>>{},execute(serial_cpu{},Op<Eval,Map, Position<Idx...>,X>{},pos, std::forward<Xs>(x)...)...);
}
template< class oP,class... X,class...Xs>
decltype(auto) execute( serial_cpu,Op<Domain,Op<oP,X...>>, Xs&&...x)
{
  return (execute(serial_cpu{},Op<Domain,X>{}, std::forward<Xs>(x)...)*...);
}


template< class Max,class...Xs>
auto execute( serial_cpu,Op<Eval,Op<Index,Max>>, Xs&&...x)
{
 // using test=typename decltype (index(execute(serial_cpu{},Op<Eval,Max>{}, std::forward<Xs>(x)...)))::te;

  return index(execute(serial_cpu{},Op<Eval,Max>{}, std::forward<Xs>(x)...));
}

template<class G, class... Xs>
auto execute(serial_cpu,Op<Eval,Map,span<>,G>,Xs&&...x)
{
  return execute(serial_cpu{},Op(Eval{},G{}), std::forward<Xs>(x)...);
}

template<class G,class Idx, class... Xs>
auto execute(serial_cpu,Op<Eval,Map,span<Idx>,G>,Xs&&...x)
{
  auto n=execute(serial_cpu{},Op<Eval,Size,span<Idx>>{}, std::forward<Xs>(x)...);

  using element_type=std::decay_t<decltype (execute(serial_cpu{},Op<Eval,Map,Position<Idx>,G>{},Position<Idx>{}, std::forward<Xs>(x)...))>;

  auto vector_out=vector_device<element_type>(n);

  for (Position<Idx> p{}; p()<n; ++p())
    vector_out[p()]=execute(serial_cpu{},Op<Eval,Map,Position<Idx>,G>{},p, std::forward<Xs>(x)...);
  return vector_field<element_type,Idx>(std::move(vector_out));
}


template<class Id,class G, class... Xs>
auto execute(serial_cpu,Op<Eval,F<Id,G>>,Xs&&...x)
{
  auto non_find_args=arguments(Op<Eval,G>{})-(variables(x)&&...);
  //using test=typename decltype (execute(serial_cpu{},Op<Eval,G>{}, std::forward<Xs>(x)...))::te;
  if constexpr (non_find_args.size==0)
  {
    auto mydomain=execute(serial_cpu{},Op<Domain,G>{},std::forward<Xs>(x)...);
    return mapu(x_i(Id{},execute(serial_cpu{},Op(Eval{},Map{},mydomain,G{}), std::forward<Xs>(x)...)));
  }
    else
    return Error{Not_found{F<Id,G>{},non_find_args}};
}









template< class...Fs,class ...Ids,class... xs,class...Xs>
auto execute( serial_cpu,Op<Eval,quimulun<Fs...>,Variables<Ids...>>, mapu<xs...>&& known_variables,Xs&&...x)
{
  auto new_variables= (execute(serial_cpu{},Op(Eval{},quimulun<Fs...>{}[Ids{}]),known_variables,std::forward<Xs>(x)...)&&...);

  if constexpr (is_Error_v<decltype(new_variables)>)
    return new_variables;
  else
  {
     auto unknowns_vars=Variables<Ids...>{}-variables(new_variables);
     auto all_variables=std::move(known_variables)&&std::move(new_variables);
     if constexpr (unknowns_vars.size==0)
       return all_variables;
     else
       return execute(serial_cpu{},Op(Eval{},quimulun<Fs...>{},unknowns_vars),std::move(all_variables),std::forward<Xs>(x)...);
  }

}




template< class...Fs,class...Xs>
auto execute( serial_cpu,Op<Eval,quimulun<Fs...>>, Xs&&...x)
{
  auto v=variables(quimulun<Fs...>{});
  return execute(serial_cpu{},Op(Eval{},quimulun<Fs...>{},v), mapu<>{}, std::forward<Xs>(x)...);
}








#endif // CUDA_TEST_CPU_H
