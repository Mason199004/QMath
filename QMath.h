#pragma once


#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <stdint.h>

#define f32 float
#define i32 int32_t




static __m128i Vec3Mask;

static inline void QMath_InitMask()
{
	Vec3Mask = _mm_setr_epi32(-1, -1, -1, 0);
}

typedef struct Vec3 {
	f32 x;
	f32 y;
	f32 z;
} vec3_t;

typedef struct Vec2 {
	f32 x;
	f32 y;
} vec2_t;

typedef struct Vec2i {
	i32 x;
	i32 y;
} vec2i_t;

typedef struct Mat4 {
	f32 m0,  m1,  m2,  m3;
	f32 m4,  m5,  m6,  m7;
	f32 m8,  m9,  m10, m11;
	f32 m12, m13, m14, m15;
} mat4_t;

static inline mat4_t QMath_Mat4_Ortho(f32 left, f32 right, f32 top, f32 bottom, f32 near, f32 far)
{
	
	mat4_t m = {
			2.0f / (right - left),            0.0f,                             0.0f,  0.0f,
			0.0f,                             2.0f / (top - bottom),            0.0f,  0.0f,
			0.0f,                             0.0f,                            -1.0f,  0.0f,
			-(right + left) / (right - left), -(top + bottom) / (top - bottom),  0.0f,  1.0f
	};
	
	return m;
}

static inline mat4_t QMath_Mat4_Translate(mat4_t m, f32 x, f32 y, f32 z)
{
	
	mat4_t m2 = {1, 0, 0, 0,
	             0, 1, 0, 0,
	             0, 0, 1, 0,
	             x, y, z, 1};

	__m128 r1 = _mm_loadu_ps(&m.m0);
	__m128 r2 = _mm_loadu_ps(&m.m4);
	__m128 r3 = _mm_loadu_ps(&m.m8);
	__m128 r4 = _mm_loadu_ps(&m.m12);

	__m128 rx1 = _mm_loadu_ps(&m2.m0);
	__m128 rx2 = _mm_loadu_ps(&m2.m4);
	__m128 rx3 = _mm_loadu_ps(&m2.m8);
	__m128 rx4 = _mm_loadu_ps(&m2.m12);

	r1 = _mm_mul_ps(r1, rx1);
	r2 = _mm_mul_ps(r2, rx2);
	r3 = _mm_mul_ps(r3, rx3);
	r4 = _mm_mul_ps(r4, rx4);

	_mm_storeu_ps(&m.m0, r1);
	_mm_storeu_ps(&m.m4, r2);
	_mm_storeu_ps(&m.m8, r3);
	_mm_storeu_ps(&m.m12, r4);

	
	return m;
}

static inline vec2_t QMath_Vec2_Add(vec2_t a, vec2_t b)
{
	
#ifndef _MSC_VER
	__m128 r1 = (__m128)_mm_loadu_si64(&a.x);
	__m128 r2 = (__m128)_mm_loadu_si64(&b.x);

	r1 = _mm_add_ps(r1, r2);

	_mm_storeu_si64(&a.x, (__m128i)r1);
#else
	a.x += b.x;
	a.y += b.y;
#endif
	
	return a;
}

static inline vec2_t QMath_Vec2_Sub(vec2_t a, vec2_t b)
{
	
#ifndef _MSC_VER
	__m128 r1 = (__m128)_mm_loadu_si64(&a.x);
	__m128 r2 = (__m128)_mm_loadu_si64(&b.x);

	r1 = _mm_sub_ps(r1, r2);

	_mm_storeu_si64(&a.x, (__m128i)r1);
#else
	a.x -= b.x;
	a.y -= b.y;
#endif
	
	return a;
}

static inline vec2_t QMath_Vec2_Mul(vec2_t a, vec2_t b)
{
	
#ifndef _MSC_VER
	__m128 r1 = (__m128)_mm_loadu_si64(&a.x);
	__m128 r2 = (__m128)_mm_loadu_si64(&b.x);

	r1 = _mm_mul_ps(r1, r2);

	_mm_storeu_si64(&a.x, (__m128i)r1);
#else
	a.x *= b.x;
	a.y *= b.y;
#endif
	
	return a;
}

static inline vec2_t QMath_Vec2_Div(vec2_t a, vec2_t b)
{
	
#ifndef _MSC_VER
	__m128 r1 = (__m128)_mm_loadu_si64(&a.x);
	__m128 r2 = (__m128)_mm_loadu_si64(&b.x);

	r1 = _mm_div_ps(r1, r2);

	_mm_storeu_si64(&a.x, (__m128i)r1);
#else
	a.x /= b.x;
	a.y /= b.y;
#endif
	
	return a;
}

static inline f32 QMath_Vec2_Cross(vec2_t a, vec2_t b)
{
	
#ifndef _MSC_VER
	__m128 r1 = (__m128)_mm_loadu_si64(&a.x);
	__m128 r2 = _mm_set_ps(0, 0, b.x, b.y);

	r1 = _mm_mul_ps(r1, r2);
	r2 = _mm_hsub_ps(r1, r1);

	_mm_storeu_si32(&a.x, (__m128i)r2);
	
	return a.x;
#else
	
	return ( a.x * b.y ) - ( b.x * a.y );
#endif
}

static inline f32 QMath_Vec2_Dot(vec2_t a, vec2_t b)
{
	
#ifndef _MSC_VER
	__m128 r1 = (__m128)_mm_loadu_si64(&a.x);
	__m128 r2 = (__m128)_mm_loadu_si64(&b.x);

	r1 = _mm_dp_ps(r1, r2, 0b00111111);

	_mm_storeu_si32(&a.x, (__m128i)r1);
	
	return a.x;
#else
	
	return ( a.x * b.y ) + ( b.x * a.y );
#endif
}

static inline vec3_t QMath_Vec3_Add(vec3_t a, vec3_t b)
{
	
	__m128 r1 = _mm_maskload_ps(&a.x, Vec3Mask);
	__m128 r2 = _mm_maskload_ps(&b.x, Vec3Mask);

	r1 = _mm_add_ps(r1, r2);
	_mm_maskstore_ps(&a.x, Vec3Mask, r1);

	
	return a;
}

static inline vec3_t QMath_Vec3_Sub(vec3_t a, vec3_t b)
{
	
	__m128 r1 = _mm_maskload_ps(&a.x, Vec3Mask);
	__m128 r2 = _mm_maskload_ps(&b.x, Vec3Mask);

	r1 = _mm_sub_ps(r1, r2);
	_mm_maskstore_ps(&a.x, Vec3Mask, r1);

	
	return a;
}

static inline vec3_t QMath_Vec3_Mul(vec3_t a, vec3_t b)
{
	
	__m128 r1 = _mm_maskload_ps(&a.x, Vec3Mask);
	__m128 r2 = _mm_maskload_ps(&b.x, Vec3Mask);

	r1 = _mm_mul_ps(r1, r2);
	_mm_maskstore_ps(&a.x, Vec3Mask, r1);

	
	return a;
}

static inline vec3_t QMath_Vec3_Div(vec3_t a, vec3_t b)
{
	
	__m128 r1 = _mm_maskload_ps(&a.x, Vec3Mask);
	__m128 r2 = _mm_maskload_ps(&b.x, Vec3Mask);

	r1 = _mm_div_ps(r1, r2);
	_mm_maskstore_ps(&a.x, Vec3Mask, r1);

	
	return a;
}

static inline f32 QMath_Vec3_Dot(vec3_t a, vec3_t b)
{
	
	__m128 r1 = _mm_maskload_ps(&a.x, Vec3Mask);
	__m128 r2 = _mm_maskload_ps(&b.x, Vec3Mask);

	r1 = _mm_dp_ps(r1, r2, 0b01111111);

	_mm_maskstore_ps(&a.x, Vec3Mask, r1);

	
	return a.x;
}

static inline vec3_t QMath_Vec3_Cross(vec3_t a, vec3_t b)
{
	
	__m128 r1 = _mm_maskload_ps(&a.x, Vec3Mask);
	__m128 r2 = _mm_maskload_ps(&b.x, Vec3Mask);

	__m128 tmp0 = _mm_shuffle_ps(r1,r1,_MM_SHUFFLE(3,0,2,1));
	__m128 tmp1 = _mm_shuffle_ps(r2,r2,_MM_SHUFFLE(3,1,0,2));
	__m128 tmp2 = _mm_mul_ps(tmp0,r2);
	__m128 tmp3 = _mm_mul_ps(tmp0,tmp1);
	__m128 tmp4 = _mm_shuffle_ps(tmp2,tmp2,_MM_SHUFFLE(3,0,2,1));
	__m128 tmp5 = _mm_sub_ps(tmp3,tmp4);

	_mm_maskstore_ps(&a.x, Vec3Mask, tmp5);

	
	return a;
}
