#ifndef CUDA_TEST_TRANSPORTER_H
#define CUDA_TEST_TRANSPORTER_H

#include "cuda_test_functions.h"
#include "cuda_runtime.h"

template<class T>
class send_internal_pointer_to_device
{

public:
  using type=T;
  explicit send_internal_pointer_to_device(const T& ){}
  type  copy_body(const type& v)const{ return v;}

  void insert_pointer_in(T& )const{}
  void remove_pointer_from(T& )const{}
 };



template<class T>
class send_to_device
{
  T const* p_;
public:
  explicit send_to_device(const T& x)
  {
    cudaMalloc(&p_,sizeof (T));
    cudaMemcpy(p_, &x, sizeof (T), cudaMemcpyHostToDevice);
  }
  __device__
  auto& operator()()const{return *p_;}

  ~send_to_device(){cudaFree(p_);}

};

template<class T>
class send_internal_pointer_to_device<vector_device<T>>
{
  T* p_;

public:
  explicit send_internal_pointer_to_device(const vector_device<T>& x)
  {
    cudaMalloc(&p_,x.byte_size());
    cudaMemcpy(p_, x.begin(),x.byte_size(), cudaMemcpyHostToDevice);
  }
  vector_device<T>  copy_body(const vector_device<T>& v)const{
    return vector_device<T>(v.size());
}
  void insert_pointer_in(vector_device<T>& v)const{
    v.begin()=p_;
  }
  void remove_pointer_from(vector_device<T>& v)const{
    v.begin()=nullptr;
  }

  ~send_internal_pointer_to_device(){
    cudaFree(p_);
  }
};


template<class T>
class send_to_device<vector_device<T>>: public send_internal_pointer_to_device<vector_device<T>>
{
  vector_device<T>  * v_;

public:
  using base=send_internal_pointer_to_device<vector_device<T>>;
  explicit send_to_device(const vector_device<T>& x): base{x},v_{nullptr}
  {
    cudaMalloc(&v_,sizeof (vector_device<T>));
    auto x_copy=base::copy_body(x);
    send_internal_pointer_to_device<vector_device<T>>::insert_pointer_in(x_copy);
    cudaMemcpy(v_, &x_copy,sizeof (vector_device<T>), cudaMemcpyHostToDevice);
    base::remove_pointer_from(x_copy);
   }

  __device__
   auto& operator()()const{
    return *v_;
  }

  ~send_to_device(){
    cudaFree(v_);
  }
};



template<class T, class Idx>
class send_internal_pointer_to_device<vector_field<T,Idx>>: public send_internal_pointer_to_device<vector_device<T>>
{
public:
  using type=vector_field<T,Idx>;
  using base=send_internal_pointer_to_device<vector_device<T>>;
  explicit send_internal_pointer_to_device(const type& x): base {x.value()}{}

  type  copy_body(const type& v)const{
  return type(base::copy_body(v.value()));
}

  void insert_pointer_in(type& v)const {
    base::insert_pointer_in(v.value());
  }
  void remove_pointer_from(type& v)const{
    base::remove_pointer_from(v.value());
  }
};


template<class T, class Idx>
class send_to_device<vector_field<T,Idx>>: public send_internal_pointer_to_device<vector_field<T,Idx>>
{
  vector_field<T,Idx>  * v_;

public:
  using type=vector_field<T,Idx>;
  using base=send_internal_pointer_to_device<vector_field<T,Idx>>;
  explicit send_to_device(const type& x): base{x},v_{nullptr}
  {
    cudaMalloc(&v_,sizeof (type));
    auto x_copy=base::copy_body(x);
    base::insert_pointer_in(x_copy);
    cudaMemcpy(v_, &x_copy,sizeof (type), cudaMemcpyHostToDevice);
    base::remove_pointer_from(x_copy);
  }

  __device__
      auto& operator()()const{return *v_;}

  ~send_to_device(){
    cudaFree(v_);
  }
};


template<class Id, class T>
class send_internal_pointer_to_device<x_i<Id,T>>: public send_internal_pointer_to_device<T>
{
public:
  using type=x_i<Id,T>;
  using base=send_internal_pointer_to_device<T>;
  explicit send_internal_pointer_to_device(const type& x): base {x()}{}

   type  copy_body(const type& v)const{
     return type(Id{},base::copy_body(v()));
}

  void insert_pointer_in(type& v)const{
    base::insert_pointer_in(v());
  }
  void remove_pointer_from(type& v)const{
    base::remove_pointer_from(v());
  }
};



template<class Id, class T>
class send_to_device<x_i<Id,T>>: public send_internal_pointer_to_device<x_i<Id,T>>
{
  x_i<Id,T>  * v_;

public:
  using type=x_i<Id,T>;
  using base=send_internal_pointer_to_device<type>;
  explicit send_to_device(const type& x): base{x},v_{nullptr}
  {
    cudaMalloc(&v_,sizeof (type));
    auto x_copy=base::copy_body(x);
    base::insert_pointer_in(x_copy);
    cudaMemcpy(v_, &x_copy,sizeof (type), cudaMemcpyHostToDevice);
    base::remove_pointer_from(x_copy);
  }

  __device__
      auto& operator()()const{return *v_;}

  ~send_to_device(){
    cudaFree(v_);
  }
};

template<class...xs>
class send_internal_pointer_to_device<mapu<xs...>>: public send_internal_pointer_to_device<xs>...
{
public:
  using type=mapu<xs...>;
  explicit send_internal_pointer_to_device(const type& x): send_internal_pointer_to_device<xs> {x[typename xs::myId{}]}...{}

  type  copy_body(const type& v)const{
    return type{send_internal_pointer_to_device<xs>::copy_body(v[typename xs::myId{}])...};
}

  void insert_pointer_in(type& v)const{
    (send_internal_pointer_to_device<xs>::insert_pointer_in(v[typename xs::myId{}]),...);
  }
  void remove_pointer_from(type& v)const{
    (send_internal_pointer_to_device<xs>::remove_pointer_from(v[typename xs::myId{}]),...);
  }
};



template<class...xs>
class send_to_device<mapu<xs...>>: public send_internal_pointer_to_device<mapu<xs...>>
{
  mapu<xs...>  * v_;

public:
  using type=mapu<xs...>;
  using base=send_internal_pointer_to_device<type>;
  explicit send_to_device(const type& x): base{x},v_{nullptr}
  {
    cudaMalloc(&v_,sizeof (type));
    auto x_copy=base::copy_body(x);
    base::insert_pointer_in(x_copy);
    cudaMemcpy(v_, &x_copy,sizeof (type), cudaMemcpyHostToDevice);
    base::remove_pointer_from(x_copy);
  }

  __device__
      auto& operator()()const{return *v_;}

  ~send_to_device(){
    cudaFree(v_);
  }
};








#endif // CUDA_TEST_TRANSPORTER_H
