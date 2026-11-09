-- FCrystalNet native: crystalnet.dll = one embedded ONNX Runtime session + JNI boundary.
-- The host is a JVM process: static CRT (/MT) across the board - an /MD-built DLL inside
-- a JVM resolves its CRT imports against the host's bundled msvcrt copy and AVs on first
-- use.
set_project("crystalnet-native")
set_xmakever("2.9.3")
set_allowedplats("windows")
set_defaultplat("windows")
set_defaultarchs("x64")
set_languages("cxx17")
add_rules("mode.debug", "mode.release")

-- Git Bash (MSYS) shells make xmake detect the mingw subsystem; force the windows
-- target explicitly, or the /MT toolchain wiring below never engages.
if is_host("windows") then
    set_plat("windows")
    set_toolchains("msvc")
end

if is_plat("windows") then
    set_runtimes("MT")
end

-- On a GBK ACP machine MSVC reads sources in the codepage by default; UTF-8 Chinese
-- comments then misdecode (C4819/C2001).
add_cxflags("/utf-8", {tools = "cl"})

option("jdk_root")
    set_default("")
    set_showmenu(true)
    set_description("JDK home, for jni.h (override: xmake f --jdk_root=D:/path/jdk-21)")
option_end()

target("crystalnet")
    set_kind("shared")
    -- native/build/bin/crystalnet.dll - the default path CrystalNetBridge.locateDll()
    -- resolves.
    set_targetdir("$(builddir)/bin")
    add_files("src/*.cpp")
    add_includedirs("src", "third_party/onnxruntime/include")
    -- Zero link dependencies: ORT is loaded at runtime (LoadLibrary on the path
    -- nativeInit receives), so nothing here can collide with MC's LWJGL or the JVM.
    -- JNI headers resolve in on_load: the description-scope sandbox in xmake 3.0.9
    -- hides os.getenv/os.exit/raise, but script scope has the full API.
    on_load(function (target)
        local jdk = get_config("jdk_root")
        if jdk == nil or jdk == "" then
            jdk = os.getenv("JAVA_HOME") or ""
        end
        if jdk == "" or not os.isfile(path.join(jdk, "include", "jni.h")) then
            raise("jni.h not resolvable - set JAVA_HOME or run `xmake f --jdk_root=<jdk>`")
        end
        target:add("includedirs", path.join(jdk, "include"),
                path.join(jdk, "include", "win32"))
    end)
target_end()
