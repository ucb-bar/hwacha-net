        .align 3
	.globl hgemm_opt_v_4_4
	.globl hgemm_opt_v_4_4_edge0
	.globl hgemm_opt_v_4_4_pre
	.globl hgemm_opt_v_4_4_post
	.globl hgemm_opt_v_4_4_edge1
	.globl hgemm_opt_v_4_4_pre_edge
	.globl hgemm_opt_v_4_4_post_edge

hgemm_opt_v_4_4_pre:
	vlh vv0, va0 # c0
	vlh vv1, va1 # c1
	vlh vv2, va2 # c2
	vlh vv3, va3 # c3
	vstop

hgemm_opt_v_4_4_pre_edge:
	vlh vv0, va0 # c0
	vstop

hgemm_opt_v_4_4:
	vlh vv4, va4                    # b0
	vfmadd.h vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.h vv1, vv4, vs5, vv1     # c1 += a10 * b0

	vlh vv5, va5                    # b1
	vfmadd.h vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.h vv3, vv4, vs13, vv3    # c3 += a30 * b0

	vlh vv6, va6                    # b2
	vfmadd.h vv0, vv5, vs2, vv0     # c0 += a01 * b1
	vfmadd.h vv1, vv5, vs6, vv1     # c1 += a11 * b1

	vlh vv7, va7                    # b3
	vfmadd.h vv0, vv6, vs3, vv0     # c0 += a02 * b2
	vfmadd.h vv1, vv6, vs7, vv1     # c1 += a12 * b2
	vfmadd.h vv0, vv7, vs4, vv0     # c0 += a03 * b3
	vfmadd.h vv1, vv7, vs8, vv1     # c1 += a13 * b3
	vfmadd.h vv2, vv5, vs10, vv2    # c2 += a21 * b1
	vfmadd.h vv3, vv5, vs14, vv3    # c3 += a31 * b1
	vfmadd.h vv2, vv6, vs11, vv2    # c2 += a22 * b2
	vfmadd.h vv3, vv6, vs15, vv3    # c3 += a32 * b2
	vfmadd.h vv2, vv7, vs12, vv2    # c2 += a23 * b3
	vfmadd.h vv3, vv7, vs16, vv3    # c3 += a33 * b3
	vstop

hgemm_opt_v_4_4_edge0:
	vlh vv4, va4                    # b0
	vfmadd.h vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.h vv1, vv4, vs5, vv1     # c1 += a10 * b0
	vfmadd.h vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.h vv3, vv4, vs13, vv3    # c3 += a30 * b0
	vstop

hgemm_opt_v_4_4_edge1:
	vlh vv4, va4                    # b0
	vfmadd.h vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vstop

hgemm_opt_v_4_4_post:
        vsh vv0, va0 # c0
        vsh vv1, va1 # c1
        vsh vv2, va2 # c2
        vsh vv3, va3 # c3
        vstop

hgemm_opt_v_4_4_post_edge:
        vsh vv0, va0 # c0
        vstop

        .globl sgemm_opt_v_4_4
	.globl sgemm_opt_v_4_4_edge0
	.globl sgemm_opt_v_4_4_pre
	.globl sgemm_opt_v_4_4_post
	.globl sgemm_opt_v_4_4_edge1
	.globl sgemm_opt_v_4_4_pre_edge
	.globl sgemm_opt_v_4_4_post_edge

sgemm_opt_v_4_4_pre:
	vlw vv0, va0 # c0
	vlw vv1, va1 # c1
	vlw vv2, va2 # c2
	vlw vv3, va3 # c3
	vstop

sgemm_opt_v_4_4_pre_edge:
	vlw vv0, va0 # c0
	vstop

sgemm_opt_v_4_4:
	vlw vv4, va4                    # b0
	vfmadd.s vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.s vv1, vv4, vs5, vv1     # c1 += a10 * b0

	vlw vv5, va5                    # b1
	vfmadd.s vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.s vv3, vv4, vs13, vv3    # c3 += a30 * b0

	vlw vv6, va6                    # b2
	vfmadd.s vv0, vv5, vs2, vv0     # c0 += a01 * b1
	vfmadd.s vv1, vv5, vs6, vv1     # c1 += a11 * b1

	vlw vv7, va7                    # b3
	vfmadd.s vv0, vv6, vs3, vv0     # c0 += a02 * b2
	vfmadd.s vv1, vv6, vs7, vv1     # c1 += a12 * b2
	vfmadd.s vv0, vv7, vs4, vv0     # c0 += a03 * b3
	vfmadd.s vv1, vv7, vs8, vv1     # c1 += a13 * b3
	vfmadd.s vv2, vv5, vs10, vv2    # c2 += a21 * b1
	vfmadd.s vv3, vv5, vs14, vv3    # c3 += a31 * b1
	vfmadd.s vv2, vv6, vs11, vv2    # c2 += a22 * b2
	vfmadd.s vv3, vv6, vs15, vv3    # c3 += a32 * b2
	vfmadd.s vv2, vv7, vs12, vv2    # c2 += a23 * b3
	vfmadd.s vv3, vv7, vs16, vv3    # c3 += a33 * b3
	vstop

sgemm_opt_v_4_4_edge0:
	vlw vv4, va4                    # b0
	vfmadd.s vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.s vv1, vv4, vs5, vv1     # c1 += a10 * b0
	vfmadd.s vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.s vv3, vv4, vs13, vv3    # c3 += a30 * b0
	vstop

sgemm_opt_v_4_4_edge1:
	vlw vv4, va4                    # b0
	vfmadd.s vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vstop

sgemm_opt_v_4_4_post:
        vsw vv0, va0 # c0
        vsw vv1, va1 # c1
        vsw vv2, va2 # c2
        vsw vv3, va3 # c3
        vstop

sgemm_opt_v_4_4_post_edge:
        vsw vv0, va0 # c0
        vstop
