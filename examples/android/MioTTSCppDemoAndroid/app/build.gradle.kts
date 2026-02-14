plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
}

val repoModelsDir = rootProject.layout.projectDirectory.dir("../../../models")
val generatedBundledAssetsRoot = layout.buildDirectory.dir("generated/bundled-assets")
val generatedBundledModelsDir = layout.buildDirectory.dir("generated/bundled-assets/models")

val syncBundledModels by tasks.registering(Sync::class) {
    from(repoModelsDir) {
        include("MioTTS-*.gguf")
        include("miocodec.gguf")
        include("wavlm_base_plus_2l_f32.gguf")
        include("*.emb.gguf")
    }
    into(generatedBundledModelsDir)
}

android {
    namespace = "com.example.miottscpp"
    compileSdk = 36

    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "com.example.miottscpp"
        minSdk = 33
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }

        externalNativeBuild {
            cmake {
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_BACKEND_DL=OFF"
                arguments += "-DGGML_CPU_ALL_VARIANTS=OFF"
                arguments += "-DGGML_CPU_KLEIDIAI=OFF"
                arguments += "-DGGML_LLAMAFILE=OFF"
                arguments += "-DGGML_VULKAN=OFF"
                arguments += "-DGGML_VIRTGPU=OFF"
                arguments += "-DGGML_HEXAGON=OFF"
                arguments += "-DGGML_ZENDNN=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=OFF"
                arguments += "-DLLAMA_BUILD_TESTS=OFF"
                arguments += "-DLLAMA_BUILD_EXAMPLES=OFF"
                arguments += "-DLLAMA_BUILD_TOOLS=OFF"
                arguments += "-DLLAMA_BUILD_SERVER=OFF"
                arguments += "-DLLAMA_LLGUIDANCE=OFF"
            }
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            isShrinkResources = false
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
        }
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(generatedBundledAssetsRoot)
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.documentfile)
    implementation(libs.material)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

tasks.named("preBuild").configure {
    dependsOn(syncBundledModels)
}
