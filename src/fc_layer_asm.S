        .text
        .globl vfc_32_pre
        .globl vfc_32
        .globl vfc_32_post
vfc_32_pre:
        vxor vv0, vs0, vs0
        vstop
vfc_32:
        vlw vv1, va1
        vfmadd.s vv0, vs1, vv1, vv0
        vstop
vfc_32_post:
        vsw vv0, va0
        vstop

        .globl vfc_encoded_32_pre
        .globl vfc_encoded_32
        .globl vfc_encoded_32_0
        .globl vfc_encoded_32_1
        .globl vfc_encoded_32_post
vfc_encoded_32_pre:
        vxor vv0, vs0, vs0
        vstop
vfc_encoded_32:
        vlb vv2, va1
        vstop
vfc_encoded_32_0:
        vand vv1, vv2, vs3
        vsll vv1, vv1, vs4
        vlxw vv1, vs1, vv1
        vfmadd.s vv0, vs5, vv1, vv0
        vstop
vfc_encoded_32_1:
        vsrl vv1, vv2, vs2
        vand vv1, vv1, vs3
        vsll vv1, vv1, vs4
        vlxw vv1, vs1, vv1
        vfmadd.s vv0, vs6, vv1, vv0
        vstop
vfc_encoded_32_post:
        vsw vv0, va0
        vstop
