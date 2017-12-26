        .text
        .align 10
        .globl vsetvlen
vsetvlen:
        vpset vp0
        vstop

        .globl vmemcpy_16
vmemcpy_16:
        vlh vv0, va0
        vsh vv0, va1
        vstop

        .globl vmemcpy_32
vmemcpy_32:
        vlw vv0, va0
        vsw vv0, va1
        vstop

        .globl vcvt_32_16
vcvt_32_16:
        vlw vv0, va0
        vfcvt.h.s vv1, vv0
        vsh vv1, va1
        vstop

        .globl vcvt_16_32
vcvt_16_32:
        vlh vv1, va0
        vfcvt.s.h vv0, vv1
        vsw vv0, va1
        vstop

        .globl vgather_16
vgather_16:
        vlw     vv0, va1 #INDEXES
        vadd    vv1, vs0, vs0
        vcmplt  vp1, vv0, vs0
@!vp1   vlxh    vv1, vs1, vv0
        vsh     vv1, va2
        vstop

        .globl vgather_32
vgather_32:
        vlw     vv0, va1 #INDEXES
        vadd    vv1, vs0, vs0
        vcmplt  vp1, vv0, vs0
@!vp1   vlxw    vv1, vs1, vv0
        vsw     vv1, va2
        vstop

        .globl vfill_16
vfill_16:
        vfcvt.h.s vs2, vs1
        vxor vv0, vs2, vs0
        vsh vv0, va0
        vstop

        .globl vfill_32
vfill_32:
        vxor vv0, vs1, vs0
        vsw vv0, va0
        vstop

        .globl vnormalize_16
vnormalize_16:
        vlh vv0, va0
        vfcvt.h.s vs5, vs3
        vfsqrt.h vs4, vs2
        vfadd.h vs4, vs4, vs5
        vfsub.h vv0, vv0, vs1
        vfdiv.h vv0, vv0, vs4
        vsh vv0, va0
        vstop

        .globl vnormalize_32
vnormalize_32:
        vlw vv0, va0
        vfsqrt.s vs4, vs2
        vfadd.s vs4, vs4, vs3
        vfsub.s vv0, vv0, vs1
        vfdiv.s vv0, vv0, vs4
        vsw vv0, va0
        vstop

        .globl vscale_16
vscale_16:
        vlh vv0, va0
        vfmul.h vv0, vv0, vs1
        vsh vv0, va0
        vstop

        .globl vscale_32
vscale_32:
        vlw vv0, va0
        vfmul.s vv0, vv0, vs1
        vsw vv0, va0
        vstop

        .globl vadd_16
vadd_16:
        vlh vv0, va0
        vfadd.h vv0, vv0, vs1
        vsh vv0, va0
        vstop

        .globl vadd_32
vadd_32:
        vlw vv0, va0
        vfadd.s vv0, vv0, vs1
        vsw vv0, va0
        vstop

        .globl vleaky_activate_16
vleaky_activate_16:
        vlh vv0, va0
        vfcvt.h.s vs3, vs2
        vfcvt.h.d vs1, vs0
        vcmpfle.h vp1, vv0, vs1
        @vp1 vfmul.h vv0, vv0, vs3
        @vp1    vsh vv0, va0
        vstop

        .globl vleaky_activate_32
vleaky_activate_32:
        vlw vv0, va0
        vfcvt.s.d vs1, vs0
        vcmpfle.s vp1, vv0, vs1
        @vp1 vfmul.s vv0, vv0, vs2
        @vp1    vsw vv0, va0
        vstop

        .globl vmax_16_init
        .globl vmax_16_iter
        .globl vmax_16_st
vmax_16_init:
        vxor vv1, vs0, vs1
        vstop

vmax_16_iter:
        vlw     vv0, va1 #INDEXES
        vxor    vv2, vs1, vs0
        vcmplt  vp1, vv0, vs0
@!vp1   vlxh    vv2, vs2, vv0
        vfmax.h vv1, vv1, vv2
        vstop

vmax_16_st:
        vsh vv1, va0
        vstop
        vredadd.h vs1, vv0
