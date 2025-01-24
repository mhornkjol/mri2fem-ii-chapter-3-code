import SVMTK as svm
from pathlib import Path


def make_choroid_plexus(lh_choroid_plexus, rh_choroid_plexus):
    lh_choroid_plexus.keep_largest_connected_component()
    rh_choroid_plexus.keep_largest_connected_component()
    lh_choroid_plexus.fill_holes()
    rh_choroid_plexus.fill_holes()

    lh_choroid_plexus.union(rh_choroid_plexus)
    choroid_plexus = svm.Surface(lh_choroid_plexus)
    choroid_plexus.fill_holes()
    choroid_plexus.repair_self_intersections()

    return choroid_plexus


if __name__ == "__main__":
    print("Start ", __file__)
    outdir = Path("mesh")
    outdir.mkdir(exist_ok=True)
    white = svm.Surface("stl_files/white.final.stl")
    lh_pial = svm.Surface("stl_files/lhpial.final.stl")
    rh_pial = svm.Surface("stl_files/rhpial.final.stl")
    bounding_surface = svm.Surface("stl_files/dura.final.stl")
    ventricles = svm.Surface("stl_files/ventricles.final.stl")
    lh_choroid_plexus = svm.Surface("stl_files/lhchoroid.plexus.final.stl")
    rh_choroid_plexus = svm.Surface("stl_files/rhchoroid.plexus.final.stl")
    choroid_plexus = make_choroid_plexus(lh_choroid_plexus, rh_choroid_plexus)
    cisterna_magna = svm.Surface("stl_files/cisterna.magna.stl")

    ventricles.enclose(choroid_plexus)
    choroid_plexus.embed(ventricles)
    ventricles.separate(choroid_plexus)

    white.enclose(ventricles)
    ventricles.embed(white)
    white.separate(ventricles)

    white.difference(cisterna_magna)

    bounding_surface.adjust_boundary(2)

    p1 = svm.Point_3(-4, -23, 3.7)
    p2 = svm.Point_3(-4, -30, -3.7)
    aqueduct = svm.Surface(ventricles)
    aqueduct.clip(p1, svm.Vector_3(p1, p2), 4.0)
    aqueduct.clip(p2, svm.Vector_3(p2, p1), 4.0)

    surfaces = [bounding_surface, lh_pial, rh_pial, white, ventricles, aqueduct, choroid_plexus]

    smap = svm.SubdomainMap(7)
    smap.add("1000000", 1)
    smap.add("*100000", 2)
    smap.add("*010000", 2)
    smap.add("*110000", 2)
    smap.add("*1000", 3)
    smap.add("*100", 4)
    smap.add("*10", 5)
    smap.add("*1", 6)

    domain = svm.Domain(surfaces, smap)
    domain.create_mesh(32)
    domain.save(str(outdir / "brain_mesh.mesh"))

    print("Finish ", __file__)
