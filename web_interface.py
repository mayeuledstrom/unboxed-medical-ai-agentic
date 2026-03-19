from matplotlib import pyplot as plt
from construct_mesh import construct_mesh

count = 0

def chat_return_image(arr, save_folder = "/home/jovyan/work/data/"):
    global count
    path = save_folder+str(count)+".png"
    count += 1
    plt.imsave(path, arr)
    res = ""
    res += f"<img src=\"{path}\"></img>"
    return res

def chat_return_obj(arr, save_folder = "/home/jovyan/work/data/"):
    global count
    path = save_folder+str(count)+".obj"
    construct_mesh(arr, path)
    count += 1
    res = """
<!-- Container -->
<div id="obj-viewer" style="width:100%; height:500px;"></div>

<!-- Import map (REQUIRED for bare imports like 'three') -->
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

    const container = document.getElementById("obj-viewer");

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);

    // Camera
    const camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(0, 2, 5);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lights
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // OBJ loader (official pattern)
    const loader = new OBJLoader();
    loader.load(
    """
    res += f'"{path}",'
    res += """
        (object) => {
            scene.add(object);
        },
        (xhr) => {
            if (xhr.total) {
                console.log((xhr.loaded / xhr.total * 100) + "% loaded");
            }
        },
        (error) => {
            console.error("OBJ load error:", error);
        }
    );

    // Resize
    window.addEventListener("resize", () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    // Animate
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
</script>
    """
    return res