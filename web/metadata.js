import { app } from "../../scripts/app.js";

function make_submenu(value, options, e, menu, node) {
    const submenu = new LiteGraph.ContextMenu(
        ["star on Github", "option 2",],
        { 
            event: e, 
            callback: function (v) { 
                if(v == "option 1"){
                    window.open("https://www.w3schools.com"); 
                }
                else{
                    alert("Not Implemented yet.")
                }
            }, 
            parentMenu: menu, 
            node:node
        }
    )
}

app.registerExtension({ 
	name: "com.blonicx.comfyui-x-rework",
	async setup() {
        const original_getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            // get the basic options 
            const options = original_getCanvasMenuOptions.apply(this, arguments);
            options.push(null); // inserts a divider
            options.push({
                content: "X-Rework",
                has_submenu: true,
                callback: make_submenu,
            })
            return options;
        }

		alert("Setup complete!")
	},
})