// SPDX-License-Identifier: Apache-2.0
/**
 * @file  RawScreen.kt
 * @brief The OpenAI JSON editor: the only path that forwards tools.
 */
package com.example.sampletestapp.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.example.quickdotai.Capability
import com.example.quickdotai.RuntimeKind
import com.example.sampletestapp.ChatController
import com.example.sampletestapp.R
import com.example.sampletestapp.RawRequest
import com.example.sampletestapp.UiState

@Composable
fun RawScreen(state: UiState, controller: ChatController, onLoadJson: () -> Unit, modifier: Modifier = Modifier) {
    var raw by rememberLocalText(state.rawRevision) { controller.state.value.rawJson }
    val parsed = remember(raw) { RawRequest.parse(raw) }
    Column(modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Raw request", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
            OutlinedButton(onClick = onLoadJson, modifier = Modifier.testTag(Tags.RAW_LOAD)) {
                Icon(painterResource(R.drawable.ic_text_file), null); Spacer(Modifier.width(6.dp)); Text("Load .json")
            }
            TextButton(onClick = controller::resetRawJson, modifier = Modifier.testTag(Tags.RAW_RESET)) { Text("Reset example") }
        }
        Text(
            "OpenAI-style JSON sent as-is. A tools array reaches the chat template, so function calling works here. The answer also appears in Chat.",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        OutlinedTextField(
            value = raw,
            onValueChange = { raw = it; controller.setRawJson(it) },
            textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
            modifier = Modifier.weight(1f).fillMaxWidth().testTag(Tags.RAW_JSON),
        )
        val d = state.descriptor
        val path = when {
            parsed is RawRequest.Parsed.Messages && d != null &&
                (Capability.MESSAGES_API in d.capabilities || d.runtime == RuntimeKind.LITERT) -> "runModelHandleWithMessagesStreaming"
            else -> "runModelHandleWithJsonStreaming"
        }
        val (ok, line) = when (parsed) {
            is RawRequest.Parsed.Messages -> true to "Valid JSON · ${parsed.messages.size} messages${if (parsed.hasTools) " · tools" else ""} · $path"
            RawRequest.Parsed.Other -> true to "Valid JSON, not a messages request · $path"
            is RawRequest.Parsed.Invalid -> false to "Invalid JSON: ${parsed.reason}"
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(painterResource(if (ok) R.drawable.ic_check else R.drawable.ic_warning), null,
                tint = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
            Spacer(Modifier.width(8.dp))
            Text(line, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Button(
            onClick = controller::sendRaw,
            enabled = ok && !state.busy,
            modifier = Modifier.fillMaxWidth().testTag(Tags.RAW_SEND),
        ) { Text("Send") }
    }
}
