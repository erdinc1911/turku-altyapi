plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

val buildNumber = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()

android {
    namespace = "com.erdinc.turkualtyapi"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.erdinc.turkualtyapi"
        minSdk = 26
        targetSdk = 35
        versionCode = buildNumber
        versionName = "0.1.$buildNumber"

        ndk {
            // Python 3.12 yalnız 64-bit destekler; tablet ve modern telefonlar arm64.
            abiFilters += listOf("arm64-v8a")
        }
    }

    signingConfigs {
        // Sabit test anahtarı: her derlemede aynı imza, böylece güncelleme üstüne kurulabilir.
        // Play Store sürümü için ayrı, gizli bir anahtar kullanılacak.
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        getByName("debug") {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

chaquopy {
    defaultConfig {
        version = "3.12"
        pip {
            install("numpy")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
}
