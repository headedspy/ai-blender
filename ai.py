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
        matrix = camera.matrix_basis
        frame = [matrix @ v for v in camera.data.view_frame(scene=scene)]
        origin = matrix.to_translation()
        frame.append(origin)
        for p in frame:
            bpy.ops.object.empty_add(location=p)
        
        # create ground plane
        if properties.ground:
            bpy.ops.mesh.primitive_plane_add(
                enter_editmode=False,
                align='WORLD',
                location=(0, 50, 0),
                size=200
            )
            

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