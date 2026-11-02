// SPDX-License-Identifier: Apache-2.0
/**
 * @file  Splash.kt
 * @brief Cold-start branding: the bolt and the Quick.AI wordmark slide
 *        in fast from the right edge, overshoot a little, then the
 *        tagline fades in and the caller swaps in the app.
 */
package com.example.quickdotaiapp.ui

import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.quickdotaiapp.R
import kotlinx.coroutines.delay

const val SPLASH_TAGLINE = "On-device AI playground"

@Composable
fun Splash(onDone: () -> Unit) {
    var entered by remember { mutableStateOf(false) }
    // 1 = parked off the right edge, 0 = home; the easing overshoots past 0.
    val slide by animateFloatAsState(
        targetValue = if (entered) 0f else 1f,
        animationSpec = tween(durationMillis = 380, easing = CubicBezierEasing(0.15f, 0.85f, 0.25f, 1.2f)),
        label = "logo",
    )
    val tagAlpha by animateFloatAsState(
        targetValue = if (entered) 1f else 0f,
        animationSpec = tween(durationMillis = 350, delayMillis = 450),
        label = "tagline",
    )
    LaunchedEffect(Unit) {
        entered = true
        delay(650)
        onDone()
    }
    val scheme = MaterialTheme.colorScheme
    Box(
        modifier = Modifier.fillMaxSize().background(scheme.surface).testTag(Tags.SPLASH),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.graphicsLayer {
                    translationX = slide * 220.dp.toPx()
                    alpha = (1f - slide).coerceIn(0f, 1f)
                },
            ) {
                Icon(
                    painter = painterResource(R.drawable.ic_bolt),
                    contentDescription = "Quick.AI lightning bolt",
                    tint = SkyBolt,
                    modifier = Modifier.size(34.dp),
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    text = buildAnnotatedString {
                        append("Quick")
                        withStyle(SpanStyle(color = scheme.primary)) { append(".AI") }
                    },
                    color = scheme.onSurface,
                    fontSize = 40.sp,
                    fontWeight = FontWeight.ExtraBold,
                    letterSpacing = (-1).sp,
                )
            }
            Spacer(Modifier.height(18.dp))
            Text(
                text = SPLASH_TAGLINE.uppercase(),
                style = MaterialTheme.typography.labelMedium,
                color = scheme.onSurfaceVariant,
                letterSpacing = 1.2.sp,
                modifier = Modifier.alpha(tagAlpha),
            )
        }
    }
}
