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
        .globl vfc_encoded_32_post
vfc_encoded_32_pre:
        vxor vv0, vs0, vs0
        vstop
vfc_encoded_32:
        vlb vv1, va1
        vlxw vv1, vs2, vv1
        vfmadd.s vv0, vs1, vv1, vv0
        vstop
vfc_encoded_32_post:
        vsw vv0, va0
        vstop
