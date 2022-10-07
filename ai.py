import bpy
import math
import random
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )       
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )

# Properties storage
class Properties(PropertyGroup):
    ground: BoolProperty(
        name="",
        description="Generate a ground plane",
        default = False
    )
    
    nrObj: IntProperty(
        name = "",
        description="Number of objects",
        default = 1,
        min = 1,
        max = 10
    )

# Create the scene
class GenerateScene(bpy.types.Operator):
    bl_idname = "wm.generate_scene"
    bl_label = "OPOPOP"
    
    def execute(self, context):
        scene = context.scene
        properties = scene.custom_properties
        
        # clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # create camera
        rnd = random.random()
        pos_z = rnd * 20.0
        rot_x = rnd 
        
        camera = bpy.ops.object.camera_add(
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, pos_z),
            rotation=(math.radians(90.0), 0, 0)
        )
        
        camera = context.object
        camera.scale = (100,100,100)
        
        #calculate camera frustum
        frustum = []
        matrix = camera.matrix_basis
        frame = [matrix @ v for v in camera.data.view_frame(scene=scene)]
        origin = matrix.to_translation()
        frame.append(origin)
        
        for p in frame:
            bpy.ops.object.empty_add(location=p)
            frustum.append(context.object)
        
        # create ground plane
        if properties.ground:
            bpy.ops.mesh.primitive_plane_add(
                enter_editmode=False,
                align='WORLD',
                location=(0, 50, 0),
                size=200
            )

        # generate nr of primitives inside the frustum
        for i in range(properties.nrObj):
            obj_type = int(random.random() * 7)
            
            if obj_type == 0:
                bpy.ops.mesh.primitive_cube_add()
            if obj_type == 1:
                bpy.ops.mesh.primitive_uv_sphere_add()
            if obj_type == 2:
                bpy.ops.mesh.primitive_ico_sphere_add()
            if obj_type == 3:
                bpy.ops.mesh.primitive_cylinder_add()
            if obj_type == 4:
                bpy.ops.mesh.primitive_cone_add()
            if obj_type == 5:
                bpy.ops.mesh.primitive_torus_add()
            if obj_type == 6:
                bpy.ops.mesh.primitive_monkey_add()

            object = context.object
            
            scene_size_x = (frustum[1].location.x * 2)
            scene_size_y = (frustum[0].location.z + (frustum[1].location.z * -1))
            scene_size_z = frustum[0].location.y
            
            random_pos_x = random.random()
            
            pos_x = ((random_pos_x * scene_size_x) + frustum[2].location.x)
            pos_y = ((random.random() * scene_size_y) + frustum[1].location.z)
            if(properties.ground and pos_y < 0):
                pos_y = 0
            
            pos_z = random_pos_x
            if pos_z > 0.5:
                pos_z = pos_z - ((pos_z-0.5)*2)
            if pos_z < 0.46:
                pos_z = pos_z + 0.46
            pos_z = ((pos_z / 0.5) * scene_size_z)
            
            object.location = (pos_x, pos_z, pos_y)
            
            object.rotation_euler = (math.radians(random.random()*180), math.radians(random.random()*180), math.radians(random.random()*180))
            
            object.scale = (random.random() * 15, random.random() * 15, random.random() * 15)

        # switch to camera perspective
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
                break
        
        # deselect all
        bpy.ops.object.select_all(action='TOGGLE')


        return {'FINISHED'}

# Generate section
class GeneratePanel(bpy.types.Panel):
    bl_label = "Generate"
    bl_idname = "PT_GeneratePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI"

    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        properties = scene.custom_properties
        
        row = layout.row()
        row.label(text = "Generate a scene for training/testing")
        row = layout.row()
        
        row = layout.row()
        
        row.prop(properties, "nrObj")
        row.prop(properties, "ground", icon = "VIEW_PERSPECTIVE")
        
        row.operator(GenerateScene.bl_idname, text = "Generate", icon = "SCENE_DATA")
        row = layout.row()
        row.label(text = (str(properties.nrObj) + " object(s)") + (" with ground" if properties.ground else ""))
        
# Evaluate section
class EvaluatePanel(bpy.types.Panel):
    bl_label = "Evaluate"
    bl_idname = "PT_EvaluatePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI"
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.label(text="Evaluate composition/colorization")
        
# ------------------------------------------------------------------------
#     Registration
# ------------------------------------------------------------------------

classes = (
    GenerateScene,
    GeneratePanel,
    Properties,
    EvaluatePanel
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    
    bpy.types.Scene.custom_properties = PointerProperty(type=Properties)
    

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    
    del bpy.types.Scene.custom_properties
    
    
if __name__ == "__main__":
    register()