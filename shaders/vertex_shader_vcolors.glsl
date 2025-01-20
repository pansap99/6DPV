#version 330

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 acolor;
        
out vec3 Normal;
out vec3 FragPos;
out vec3 vColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
 
        FragPos = vec3(model * vec4(pos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal; 
        vColor = acolor;
        gl_Position = projection * view * vec4(FragPos, 1.0);
}