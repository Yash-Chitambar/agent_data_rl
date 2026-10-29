"""Photo-referenced visual dressing; all geoms are massless and collision-free.

Metric task geometry and robot transforms remain owned by pc_bench/builder.
Decorations are intentionally procedural so the same assets work in MuJoCo Warp.
"""
import math
import mujoco
import numpy as np

SILVER = (.64, .65, .66, 1)
BLACK = (.025, .028, .030, 1)


def add_details(sp, variant, pc_profile=None):
    def box(body, name, pos, half, color, **kw):
        return body.add_geom(name='detail_' + name, type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=pos, size=half, rgba=color, contype=0, conaffinity=0,
                             mass=0, **kw)

    def rounded(body, name, pos, width, length, thick, radius, color):
        # Rounded rectangle in x/z, extruded in y (up for the PC and reader).
        verts = []
        for y in [-thick/2, thick/2]:
            for cx, cz, start in [(width/2-radius,length/2-radius,0),
                                  (-width/2+radius,length/2-radius,90),
                                  (-width/2+radius,-length/2+radius,180),
                                  (width/2-radius,-length/2+radius,270)]:
                for a in np.linspace(start, start+90, 7):
                    a = math.radians(a)
                    verts.append([cx+radius*math.cos(a), y, cz+radius*math.sin(a)])
        mesh = 'detail_mesh_' + name
        sp.add_mesh(name=mesh, uservert=np.asarray(verts).ravel().tolist())
        body.add_geom(name='detail_'+name, type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh, pos=pos, rgba=color, contype=0, conaffinity=0, mass=0)

    # Broad indoor illumination; no hard spotlight shadows in the reference lab.
    sp.visual.headlight.ambient = [.40,.40,.40]
    sp.visual.headlight.diffuse = [.40,.40,.40]
    for light in sp.lights:
        light.castshadow = False
        light.ambient = [.18,.18,.18]
        light.diffuse = [.45,.45,.45]
    # The bench is plain dark laminate. No material: MuJoCo Warp renders a material's rgba and
    # ignores geom_rgba, which disables the per-world table colour randomization (PickEnv).
    fix = next(b for b in sp.bodies if b.name == 'fixture')
    # Case rim/lid and rubber plinth; the physical rear panel is left open at the target.
    for g in sp.geoms:
        if g.name.startswith('fixbox_'):
            g.rgba = SILVER
    # Both Type-C openings use the original task dimensions, never photo scaling.
    from blind_insertion.assets.builder import USBC_REC, USBC_TONGUE
    next(g for g in fix.geoms if g.name == 'vis_usbc_1').rgba = [0, 0, 0, 0]

    def stadium(w, h):
        r = h / 2
        return np.array([(cx + r * math.cos(a), r * math.sin(a))
                         for cx, angles in [(w/2-r, np.linspace(-math.pi/2, math.pi/2, 13)),
                                            (-w/2+r, np.linspace(math.pi/2, 3*math.pi/2, 13))]
                         for a in angles])

    inner = stadium(*USBC_REC)
    outer = stadium(USBC_REC[0] + .0008, USBC_REC[1] + .0008)
    for label, offset in [('usbc_1', -.015), ('usbc_2_target', 0.)]:
        # Individual ring segments leave the aperture open (a single convex mesh would fill it).
        for i in range(len(inner)):
            j = (i+1) % len(inner)
            verts = [[x+offset,y,z] for z in [.00004,.00022]
                     for x,y in [inner[i],inner[j],outer[j],outer[i]]]
            name = f'{label}_rim_{i}'
            sp.add_mesh(name=name, uservert=np.array(verts).ravel().tolist())
            fix.add_geom(name='detail_'+name, type=mujoco.mjtGeom.mjGEOM_MESH,
                         meshname=name, rgba=[.35,.36,.36,1], mass=0, contype=0, conaffinity=0)
        verts = [[x+offset,y,z] for z in [-.0004,-.0003] for x,y in inner]
        name = label+'_interior'
        # The first port is a visual reference; the second retains the original collision cavity.
        if offset:
            verts = [[x,y,z+.00065] for x,y,z in verts]
        sp.add_mesh(name=name, uservert=np.array(verts).ravel().tolist())
        fix.add_geom(name='detail_'+name,type=mujoco.mjtGeom.mjGEOM_MESH,
                     meshname=name,rgba=BLACK,mass=0,contype=0,conaffinity=0)
        box(fix,label+'_tongue',(offset,0,.00038),
            (USBC_TONGUE[0]/2,USBC_TONGUE[1]/2,.00004),(.08,.085,.09,1))
    sp.geom('fixbox_0').rgba = [0,0,0,0]
    if pc_profile is None:
        rounded(fix,'pc_body',(.0215,-.004,-.053),.103,.090,.038,.007,SILVER)
        rounded(fix,'pc_lid',(.0215,.0175,-.049),.103,.098,.002,.008,SILVER)
        rounded(fix,'pc_seam',(.0215,.0145,-.049),.102,.097,.001,.007,BLACK)
        rounded(fix,'pc_foot',(.0215,-.024,-.049),.099,.094,.002,.007,BLACK)
    else:
        values=pc_profile['pc']['measurements']
        w,h,depth=[float(values[k]['value']) for k in ('width','body_height','depth')]
        feet=float(pc_profile['pc']['pad_height_derived']['value'])
        from blind_insertion.scenes.pc_rear import layout
        rear=layout(pc_profile)
        x=w/2-rear['target_from_left'];bottom=-rear['port_height']+feet;top=bottom+h
        # Recessed main case leaves the connector panel and physical opening exposed.
        rounded(fix,'pc_body',(x,bottom+h/2,-(depth+.0045)/2),w,depth-.0045,h-.002,.007,SILVER)
        rounded(fix,'pc_lid',(x,top-.001,-depth/2),w,depth,.002,.008,SILVER)
        rounded(fix,'pc_seam',(x,top-.003,-depth/2),w-.001,depth-.001,.001,.007,BLACK)
    for i in range(19):
        box(fix,f'vent{i}',(-.021+i*.0046,.012,-.0002),(.0013,.00065,.0005),BLACK)
    # Rear-panel metal rims, dark inserts and connector tongues.
    for name,x,y,w,h in [('usb_a',.013,0,.0045,.012),
                        ('rj45',.028,-.002,.016,.013),('hdmi',.050,.006,.015,.0045),
                        ('dp',.050,-.006,.016,.0045),('lock',-.015,-.013,.006,.002)]:
        box(fix,name+'_rim',(x,y,.00032),(w/2+.00045,h/2+.00045,.0003),(.43,.44,.43,1))
        box(fix,name+'_cavity',(x,y,.00066),(w/2,h/2,.00005),BLACK)
        if name != 'lock':
            box(fix,name+'_tongue',(x,y-.0002,.00074),(w*.34,h*.13,.00005),(.12,.13,.12,1))
    for i in range(8):
        box(fix,f'ethernet_pin{i}',(.0225+i*.0015,.002,.0008),(.00016,.0014,.00008),(.62,.50,.22,1))
    for x in [.022,.034]:
        box(fix,'led'+str(x),(x,-.007,.0008),(.001,.0006,.0001),(.32,.39,.17,1))
    # Small top stickers visible in photos (color blocks, no invented fine text).
    for name,x,col in [('ryzen',-.004,(.18,.18,.17,1)),('radeon',.014,(.55,.08,.06,1))]:
        box(fix,name,(x,.0187,-.013),(.008,.00015,.007),col)
        for i in range(3):
            box(fix,name+str(i),(x,.0189,-.010-i*.002),(.005-i*.001,.00005,.00025),(.75,.75,.72,1))

    peg = next(b for b in sp.bodies if b.name == 'peg')
    c, half = variant.peg_boxes[0]
    bw, bt, bl = np.array(half)*2
    rounded(peg,'reader_shell',(c[0],c[1]-.0003,c[2]-.0005),
            bw,bl-.001,bt-.0006,.004,BLACK)
    silver_w = float(pc_profile['reader']['measurements']['body_width']['value']) if pc_profile else bw-.0014
    silver_l = float(pc_profile['reader']['measurements']['body_length']['value']) if pc_profile else bl-.003
    rounded(peg,'reader_silver',(0,bt/2-.00025,c[2]-.001),silver_w,silver_l,.0005,.004,(.70,.71,.70,1))
    # SD opening at the paddle's distal end, green board and recessed black slit.
    end=c[2]+bl/2-.0005
    box(peg,'sd_rim',(0,0,end),(bw*.42,bt*.28,.0002),(.31,.32,.31,1))
    box(peg,'sd_slot',(0,0,end+.00025),(bw*.39,bt*.17,.0001),BLACK)
    box(peg,'sd_board',(0,-bt*.12,end+.00036),(bw*.37,.00045,.00006),(.03,.22,.14,1))
    # Neck remains the physical mesh; add longitudinal rubber seam.
    box(peg,'neck_seam',(0,variant.handle['height']/2-.00012,
                          .00665+variant.handle_height/2),
        (variant.handle['width']*.43,.00012,variant.handle_height*.45),(.08,.085,.08,1))

    if pc_profile is not None:
        top=-rear['port_height']+float(pc_profile['pc']['pad_height_derived']['value'])+float(pc_profile['pc']['measurements']['body_height']['value'])
        for g in fix.geoms:
            if g.name.startswith(('detail_ryzen','detail_radeon')):
                g.pos[1] += top-.0185

    world=sp.worldbody
    next(g for g in sp.geoms if g.name == 'table').rgba = [.075,.08,.078,1]   # near black, as in the lab video
    # Workbench backdrop: wall, outlet raceway, tool board, tools, power strip and monitors.
    # Lit lab room around the bench (lab photos): light grey walls, ceiling, grey floor (the bench
    # top is z = 0, the floor 0.75 m below), dark side benches, so no camera sees empty space.
    WALL = (.80, .81, .82, 1)
    box(world,'wall',(.83,.31,.50),(.025,3.0,1.30),WALL)
    box(world,'room_wall_right',(-.60,-2.20,.50),(1.45,.025,1.30),WALL)
    box(world,'room_wall_left',(-.60,2.80,.50),(1.45,.025,1.30),WALL)
    box(world,'room_wall_back',(-2.05,.31,.50),(.025,3.0,1.30),WALL)
    box(world,'room_ceiling',(-.60,.31,1.80),(1.45,3.0,.02),(.92,.92,.93,1))
    box(world,'room_floor',(-.60,.31,-.77),(1.45,3.0,.02),(.38,.38,.40,1))
    for i,y in enumerate([-.55,-.95]):
        box(world,f'room_ceiling_panel{i}',(.10,y+1.10,1.775),(.30,.15,.004),(1,1,1,1))
    box(world,'room_bench_right',(.35,-.75,-.01),(.30,.38,.01),(.06,.062,.065,1))
    box(world,'room_bench_left',(.35,1.37,-.01),(.30,.38,.01),(.06,.062,.065,1))
    box(world,'room_mat',(.42,-.62,.002),(.16,.12,.002),(.15,.45,.75,1))
    box(world,'room_box_left',(.40,1.25,.03),(.05,.07,.03),(.85,.85,.83,1))
    box(world,'raceway',(.798,.31,.09),(.012,1.20,.045),(.74,.75,.76,1))
    box(world,'pegboard',(.795,.31,.67),(.013,.68,.20),(.72,.58,.42,1))
    for y in np.arange(-.32,.96,.045):
        for z in np.arange(.51,.85,.045):
            box(world,f'hole{y:.3f}_{z:.3f}',(.780,y,z),(.0005,.002,.008),(.16,.14,.11,1))
    for i,y in enumerate([-.18,.36,.86]):
        box(world,f'outlet{i}',(.783,y,.09),(.004,.026,.028),(.38,.39,.37,1))
        for dy in [-.011,.011]:
            box(world,f'socket{i}_{dy}',(.778,y+dy,.09),(.001,.003,.009),BLACK)
    box(world,'power_strip',(.58,-.20,.019),(.026,.13,.018),(.08,.085,.08,1))
    for i in range(6):
        for dx in [-.009,.009]:
            box(world,f'power_slot{i}_{dx}',(.58+dx,-.30+i*.039,.0375),(.002,.006,.0003),BLACK)
    box(world,'switch',(.58,-.315,.038),(.012,.008,.001),(.43,.06,.04,1))
    box(world,'bit_case',(.755,.58,.43),(.018,.070,.10),(.47,.51,.055,1))
    box(world,'bit_inset',(.735,.58,.44),(.002,.057,.077),(.21,.23,.13,1))
    for i in range(10):
        box(world,f'bit{i}',(.732,.53+i*.011,.44),(.002,.002,.05),(.44,.46,.44,1))
    box(world,'hammer_handle',(.750,.21,.44),(.012,.012,.14),(.10,.095,.08,1))
    box(world,'hammer_stripe',(.736,.21,.44),(.001,.004,.11),(.56,.23,.06,1))
    box(world,'hammer_head',(.75,.21,.59),(.025,.058,.022),(.32,.34,.33,1))
    for i,y in enumerate([-.49,1.10]):
        box(world,f'monitor{i}',(.50,y,.21),(.022,.20,.16),BLACK)
        box(world,f'screen{i}',(.476,y,.21),(.001,.188,.148),(.035,.045,.05,1))
    # Curved cables approximated by short capsules, on the table behind the task.
    for cable in range(3):
        pts=[(.61+.035*math.sin(t*math.pi*2+cable),-.32+t*.75,.012+cable*.003) for t in np.linspace(0,1,24)]
        for i,(a,b) in enumerate(zip(pts,pts[1:])):
            world.add_geom(name=f'detail_cable{cable}_{i}',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[*a,*b],size=[.0018,0,0],rgba=BLACK,contype=0,conaffinity=0,mass=0)
