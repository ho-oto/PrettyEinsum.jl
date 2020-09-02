using PrettyEinsum
using Test

using TensorOperations

@testset "PrettyEinsum.jl" begin
    rsl1 = zeros(ComplexF64, 5, 5, 2)
    rsl2 = zeros(ComplexF64, 5, 5, 2)
    rsl4 = zeros(ComplexF64, 5, 5, 2)

    l = randn(ComplexF64, 5, 5)
    a = randn(ComplexF64, 5, 5, 2)
    r = randn(ComplexF64, 5, 5)
    H = randn(ComplexF64, 2, 2, 2, 2)

    @tensoropt !(p, p_lower_r, p_upper_l, p_upper_r) rsl1[l, r, p] =
        l[l, L] *
        a[L, CL, p_upper_l] *
        a[CL, CR, p_upper_r] *
        r[CR, R] *
        conj(a[r, R, p_lower_r]) *
        H[p, p_lower_r, p_upper_l, p_upper_r]

    t, _ = @optimalcontractiontree !(p, p_lower_r, p_upper_l, p_upper_r) rsl1[l, r, p] =
        l[l, L] *
        a[L, CL, p_upper_l] *
        a[CL, CR, p_upper_r] *
        r[CR, R] *
        conj(a[r, R, p_lower_r]) *
        H[p, p_lower_r, p_upper_l, p_upper_r]
    @test t == [[1, [6, 2]], [5, [3, 4]]]

    # inplace
    peinp!"""
    diagram:
    #      ______[2|l,r,p]____(r@2=l@3)____[3|l,r,p]_
    #     |        |                         |       |
    # (r@1=l@2)  (p@2=p_up_L@6)   (p@3=p_up_R@6) (r@3=l@4)
    #   [1|l,r]  [6|p_lo_L,p_lo_R,p_up_L,p_up_R]   [4|l,r]
    #    l@1     p_lo_L@6         (p@5=p_lo_R@6) (r@4=r@5)
    #                                        |       |
    #                               l@5___[5*|l,r,p]_| -> [>|l@1,l@5,p_lo_L@6]
    contraction: [((2,6),1),((3,4),5)]
    """(
        rsl2,
        l,
        a,
        a,
        r,
        a,
        H,
    )
    #=
    # allocate
    rsl3 = peinew"""
    diagram:
    #      ______[2|l,r,p]____(r@2=l@3)____[3|l,r,p]_
    #     |        |                         |       |
    # (r@1=l@2)  (p@2=p_up_L@6)   (p@3=p_up_R@6) (r@3=l@4)
    #   [1|l,r]  [6|p_lo_L,p_lo_R,p_up_L,p_up_R]   [4|l,r]
    #    l@1     p_lo_L@6         (p@5=p_lo_R@6) (r@4=r@5)
    #                                        |       |
    #                               l@5___[5*|l,r,p]_| -> [>|l@1,l@5,p_lo_L@6]
    contraction: [((2,6),1),((3,4),5)]
    """(
        l,
        a,
        a,
        r,
        a,
        H,
    )

    foo!(x, A, L, R, h) = @peinp! """
    diagram:
    #       __[$A@2|l,r,p]____(r@2=l@3)____[$A@3|l,r,p]__
    #      |       |                            |        |
    #  (r@1=l@2) (p@2=p_up_L@6)      (p@3=p_up_R@6)  (r@3=l@4)
    # [$L@1|l,r] [$h@6|p_lo_L,p_lo_R,p_up_L,p_up_R] [$R@4|l,r]
    #     l@1    p_lo_L@6            (p@5=p_lo_R@6)  (r@4=r@5)
    #                                           |        |
    #                               l@5___[$A@5*|l,r,p]__| -> [$x@>|l@1,l@5,p_lo_L@6]
    contraction: [((2,6),1),((3,4),5)]
    """
    foo!(rsl4, a, l, r, H)

    foo(A, L, R, h) = @peinew """
    diagram:
    #       __[$A@2|l,r,p]____(r@2=l@3)____[$A@3|l,r,p]__
    #      |       |                            |        |
    #  (r@1=l@2) (p@2=p_up_L@6)      (p@3=p_up_R@6)  (r@3=l@4)
    # [$L@1|l,r] [$h@6|p_lo_L,p_lo_R,p_up_L,p_up_R] [$R@4|l,r]
    #     l@1    p_lo_L@6            (p@5=p_lo_R@6)  (r@4=r@5)
    #                                           |        |
    #                               l@5___[$A@5*|l,r,p]__| -> [>|l@1,l@5,p_lo_L@6]
    contraction: [((2,6),1),((3,4),5)]
    """
    rsl5 = foo(a, l, r, H)
    =#
    # @test rsl1 == rsl2 == rsl3 == rsl4 = rsl5
    @test rsl1 ≈ rsl2
end
