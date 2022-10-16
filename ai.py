import bpy
import math
import bgl
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
    
    groundBias: FloatProperty(
        name = "",
        description = "Ground Bias",
        default = 0,
        min = 0,
        max = 1
    )
    
    rndGroundBias: BoolProperty(
        name="",
        description="Randomize the Ground Bias",
        default = False
    )
    
    ratings: EnumProperty(
        items=[
            ('1', '1', 'Terrible', '', 0),
            ('2', '2', 'Bad', '', 1),
            ('3', '3', 'Passable', '', 2),
            ('4', '4', 'Good', '', 3),
            ('5', '5', 'Amazing', '', 4)
        ],
        default='3'
    )
    
    filepath: StringProperty(
        name = "Path",
        description = "Path to the folder containing the files to import",
        default = "",
        subtype = 'DIR_PATH'
    )

class Constants():
    #minimal distance from camera
    min_z = 50
    
    #max object scale on an axis
    max_scale = 5

# output messagebox function
def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)

# Create the scene
class GenerateScene(bpy.types.Operator):
    bl_idname = "wm.generate_scene"
    bl_label = "Scene Generation"
    
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
        
        camera.data.lens = 32
        
        # create ground plane
        if properties.ground:
            bpy.ops.mesh.primitive_plane_add(
                enter_editmode=False,
                align='WORLD',
                location=(0, 50, 0),
                size=200
            )

        # generate nr of primitives
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
            
            #calculate a random point within the camera frustum
            scene_size_x = (frustum[1].location.x * 2)
            scene_size_y = (frustum[0].location.z + (frustum[1].location.z * -1))
            scene_size_z = frustum[0].location.y
            
            pos_z = random.random() * scene_size_z + Constants.min_z
            if pos_z > scene_size_z:
                pos_z = scene_size_z
            
            calc_x = (pos_z * (scene_size_x/2)) / (scene_size_z)
            pos_x = random.random() * calc_x * (-1 if random.random() >= 0.5 else 1)
            
            # include ground bias
            if properties.rndGroundBias:
                properties.groundBias = random.random()
            
            if properties.ground and random.random() < properties.groundBias:
                pos_y = 0
            else:
                calc_y = (pos_z * (scene_size_y/2)) / (scene_size_z)
                pos_y = random.random() * calc_y
            
            object.location = (pos_x, pos_z, pos_y)
            
            object.rotation_euler = (math.radians(random.randint(1, 8)*45), 
                                     math.radians(random.randint(1, 8)*45), 
                                     math.radians(random.randint(1, 8)*45))
            
            object.scale = (random.random() * Constants.max_scale + 1, 
                            random.random() * Constants.max_scale + 1, 
                            random.random() * Constants.max_scale + 1)

        # switch to camera perspective
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
                break
        
        # deselect all
        bpy.ops.object.select_all(action='TOGGLE')
        
        # turn off overlays
        bpy.context.space_data.overlay.show_overlays = False

        return {'FINISHED'}



class SaveScene(bpy.types.Operator):
    bl_idname = "wm.save_scene"
    bl_label = "OPOPOP"
    

    
    def execute(self, context):
        scene = context.scene
        properties = scene.custom_properties
        
        # check if filepath is set
        if not properties.filepath:
            ShowMessageBox("Error: Filepath field is empty", "No Filepath Set", 'ERROR')
            return{'CANCELLED'}
        
        # render viewport and save image
        sce = bpy.context.scene.name
        bpy.ops.render.opengl(write_still=True)
        bpy.data.images["Render Result"].save_render(filepath=bpy.path.abspath(properties.filepath) + 
                                                    str(properties.ratings) + "_" +
                                                    (("0"+str(properties.nrObj)) if properties.nrObj != 10 else (str(properties.nrObj))) +
                                                    ("_G" if properties.ground else "_N") +
                                                    ".png"
                                                     )
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
        row.label(text = (str(properties.nrObj) + " object(s)") + (" with ground" if properties.ground else ""))
        row = layout.row()
        
        row = layout.row()
        
        row.prop(properties, "nrObj")
        row.prop(properties, "ground", icon = "VIEW_PERSPECTIVE")
        
        row.operator(GenerateScene.bl_idname, text = "Generate", icon = "SCENE_DATA")
        
        if properties.ground:
            row = layout.row()
            row.prop(properties, "rndGroundBias", icon = "OUTLINER_DATA_LATTICE")
            row.prop(properties, "groundBias", slider = True)
        
        row = layout.row()
        
        
        row.prop(properties, 'ratings', expand=True)
        row.operator(SaveScene.bl_idname, text = "Save")
        row = layout.row()
        row.prop(properties, 'filepath')
        
        
        
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
    SaveScene,
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