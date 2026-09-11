extends SceneTree

func _initialize():
    var packer = PCKPacker.new()
    var result = packer.pck_start("res://dist/STS2AgentOverlay.pck")
    if result == OK:
        result = packer.add_file("res://mod_manifest.json", "res://mod_manifest.json")
    if result == OK:
        result = packer.flush()
    quit(0 if result == OK else 1)
