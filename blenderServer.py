import bpy
import json
import math
import asyncio
from asyncio import coroutine, sleep, Task, wait_for
import sys
import os
sys.path.append(os.path.dirname(__file__))
import aiohttp
import asyncio_bridge
import feedparser
from asyncio_bridge import BlenderListener
from mathutils import Vector

def center():
    for i in range(56):
        bpy.data.objects[2+i].location.x -= bpy.data.objects[2+i].dimensions.x/2
        bpy.data.objects[2+i].location.y += bpy.data.objects[2+i].dimensions.y/2
        bpy.data.objects[2+i].location.z -= bpy.data.objects[2+i].dimensions.z/2

def get_view(area,a,b,z,z_look,n):
    r = 0.75*(1+a)
    theta = math.pi*b
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    moveCamera(area,x,y,z,0.5*z_look)
    do_render(area,n)

def do_render(area,n):
    bpy.context.scene.render.filepath='/Volumes/RAM Disk/renders/m'+str(n)+'.png'
    obj = bpy.data.objects[n+2]
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    ctx=bpy.context.copy()
    ctx['area']=area
    bpy.ops.object.hide_render_clear(ctx)    
    bpy.ops.render.render(use_viewport=True,write_still=True)
    bpy.ops.object.hide_render_set(ctx)

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def moveCamera(area,x,y,z,z_look):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Camera'].select = True
    ctx=bpy.context.copy()
    ctx['area']=area
    bpy.ops.transform.translate(ctx,value=Vector((x,y,z)) - bpy.data.objects['Camera'].location)
    look_at(bpy.data.objects['Camera'],Vector((0.0,0.0,z_look)))
    

def callMethod(method,params,area):
    if method == "moveCamera":
        moveCamera(area,params[0],params[1],params[2])
    elif method == "get_view":
        get_view(area,params[0],params[1],params[2],params[3],params[4]) 

@coroutine
def http_server():
    import aiohttp
    from aiohttp import web

    ctx_3d = {}

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx_3d = area
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            override=bpy.context.copy()
            override['area']=area
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects['Lamp'].select = True
            bpy.ops.object.hide_render_set(override,unselected=True)
            bpy.data.objects['Lamp'].select = False

    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128
    bpy.context.scene.render.image_settings.file_format='PNG'
    bpy.context.scene.render.image_settings.color_mode='BW'

    @coroutine
    def handle(request):
        data = yield from request.text()
        obj = json.loads(data)
        callMethod(obj['method'],obj['params'],ctx_3d)
        return web.Response(text=json.dumps({'jsonrpc':'2.0','id':obj['id'],'result':'GOOD WORK!'}))


    @coroutine
    def init(loop):
        app = web.Application(loop=loop)
        app.router.add_route('POST', '/', handle)

        srv = yield from loop.create_server(app.make_handler(),
                                            '127.0.0.1', 9090)
        return srv
    yield from init(asyncio.get_event_loop())

if __name__ == "__main__":
    asyncio_bridge.register()
    bpy.ops.bpy.start_asyncio_bridge()
    Task(http_server())
