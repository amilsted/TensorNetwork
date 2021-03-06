// Copyright 2019 The TensorNetwork Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

Vue.component(
    'edge',
    {
        mixins: [mixinGet, mixinGeometry],
        props: {
            edge: Array,
            state: Object
        },
        computed: {
            node1: function() {
                return this.getNode(this.edge[0][0]);
            },
            node2: function() {
                return this.getNode(this.edge[1][0]);
            },
            angle1: function() {
                return this.axisAngle(this.edge[0][1], this.node1.axes.length)
                    + this.node1.rotation;
            },
            angle2: function() {
                return this.axisAngle(this.edge[1][1], this.node2.axes.length)
                    + this.node2.rotation;
            },
            x1: function() {
                return this.node1.position.x + this.axisX(this.angle1);
            },
            y1: function() {
                return this.node1.position.y + this.axisY(this.angle1);
            },
            x2: function() {
                return this.node2.position.x + this.axisX(this.angle2);
            },
            y2: function() {
                return this.node2.position.y + this.axisY(this.angle2);
            }
        },
        template: `
            <line class="edge" :x1="x1" :y1="y1" :x2="x2" :y2="y2"
                stroke="#ddd" stroke-width="5" stroke-linecap="round" />
        `
    }
);

Vue.component(
    'proto-edge',
    {
        mixins: [mixinGeometry],
        props: {
            x: Number,
            y: Number,
            node: Object,
            axis: Number,
        },
        computed: {
            angle: function() {
                return this.axisAngle(this.axis, this.node.axes.length)
                    + this.node.rotation;
            },
            x0: function() {
                return this.node.position.x + this.axisX(this.angle);
            },
            y0: function() {
                return this.node.position.y + this.axisY(this.angle);
            }
        },
        template: `
            <line class="edge" :x1="x0" :y1="y0" :x2="x" :y2="y"
                stroke="#bbb" stroke-width="5" stroke-linecap="round" />
        `
    }
);
