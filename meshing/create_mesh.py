

def repair_surface(surface, edge_length=1.0):
    # Keeps the largest connected component.
    surface.keep_largest_connected_component() 
    # Fills most holes in triangulation, not topological.
    surface.fill_holes()
    # Remeshing surface to specified edge length
    surface.isotropic_remeshing(edge_length, 5, False)
    # Volume perserving surface smoothing.
    surface.smooth_taubin(20)     
    # Separate narrow gaps to avoid bridges in meshing.
    surface.separate_narrow_gaps(.4,.4) 
    # Removes self-intersection       
    surface.repair_self_intersections(volume_threshold=0.4,
                                      cap_threshold=170,
                                      needle_threshold=1.5,
                                      collapse_threshold=0.4)   
    return surface

if __name__ == "__main__":
    print("Start ", __file__)
    import SVMTK as svm
    from pathlib import Path
    outdir = Path("mesh")
    outdir.mkdir(exist_ok=True)
    
    import argparse
    import time 
    parser = argparse.ArgumentParser()
    parser.add_argument('--rhpial',         default="../stl_files/rhpial.final.stl", type=str)
    parser.add_argument('--lhpial',         default="../stl_files/lhpial.final.stl", type=str)
    parser.add_argument('--white',          default="../stl_files/white.final.stl", type=str)    
    parser.add_argument('--dura',           default="../stl_files/dura.final.stl", type=str)
    parser.add_argument('--ventricles',     default="../stl_files/ventricles.final.stl", type=str)
    parser.add_argument('--foramen_magnum', default="../stl_files/foramen-magnum.stl", type=str)    
    parser.add_argument('--rhchoroid',      default="../stl_files/rhchoroid.plexus.final.stl", type=str)        
    parser.add_argument('--lhchoroid',      default="../stl_files/lhchoroid.plexus.final.stl", type=str)         
    parser.add_argument('--resolution'    , default=84, type=float)

    Z = parser.parse_args()
    white   = svm.Surface(Z.white) 
    lh_pial = svm.Surface(Z.lhpial)
    rh_pial = svm.Surface(Z.rhpial)

    bounding_surface  = svm.Surface(Z.dura)
    ventricles  = svm.Surface(Z.ventricles)
    #Create a copy of the ventricles 
    aqueduct    = svm.Surface(Z.ventricles)
    # Adjust the edge length.
    aqueduct.isotropic_remeshing(0.707,5,False)

    # Load left choiroid plexus and repair it.
    lhcp = svm.Surface(Z.lhchoroid)
    lhcp = repair_surface(lhcp,0.707)       
    # Load left choiroid plexus and repair it.
    rhcp = svm.Surface(Z.rhchoroid)
    rhcp = repair_surface(rhcp,0.707)    

    bounding_surface.adjust_boundary(1.0)
    bounding_surface = repair_surface(bounding_surface,1.0)    

    foramen_magnum = svm.Surface(Z.foramen_magnum)
    # Clips the bounding and white surface.
    bounding_surface.clip(foramen_magnum,True)
    white.clip(foramen_magnum,True)    
        
    # Locate points close to the preferred cut.
    p1 = svm.Point_3(-7.0, -23.0 , 4.0)
    p2 = svm.Point_3(-6.9, -30.2 , -5.6)
    
    # Get clips perpendicular to the centerline. 
    clp1 = aqueduct.get_perpendicular_cut(p1,.0) 
    clp2 = aqueduct.get_perpendicular_cut(p2,.0 )
    
    # Clips the ventricles system so that only cerebral aqueduct remains.
    aqueduct.clip(clp1, invert=True )
    aqueduct.clip(clp2) 

     
    # Set the structure of the surfaces.
    surfaces = [bounding_surface, lh_pial, rh_pial, white, ventricles, lhcp, rhcp, aqueduct]

    # ----- Defining the SubdomainMap -----
    smap = svm.SubdomainMap(len(surfaces))  
    # Sets the subdomain tags based on  
    smap.add("10000000", 1)
    smap.add("11000000", 2)
    smap.add("10100000", 2)
    smap.add("*10000", 3)
    smap.add("*1000", 4)
    smap.add("*100", 5) 
    smap.add("*010", 5)    
    smap.add("*1", 6)  

    # Domain object
    domain = svm.Domain(surfaces, smap)
    # ----- Preserve sharp edges -----
    # Detect and preserve sharp edges in a given plane.
    domain.add_sharp_border_edges(white, foramen_magnum, 60)
    # Detect and preserve sharp edges in a given plane.
    domain.add_sharp_border_edges(bounding_surface,foramen_magnum,60)    
    
    domain.add_sharp_border_edges(aqueduct,0)    
    
    domain.create_mesh(Z.resolution)
    

    domain.save(str(outdir / "gonzo.mesh"))

    print("Finish ", __file__)
