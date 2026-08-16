export function getAuthToken(): string {
  const local =
    localStorage.getItem("token") ||
    localStorage.getItem("authToken") ||
    localStorage.getItem("auth_token");
  if (local) return local;

  const cookieMatches = document.cookie
    .split("; ")
    .filter((row) => row.startsWith("authToken=") || row.startsWith("token=") || row.startsWith("auth_token="));

  if (cookieMatches.length > 0) {
    return cookieMatches[0].split("=")[1] || "";
  }
  return "";
}

export function setAuthToken(token: string): void {
  localStorage.setItem("token", token);
  localStorage.setItem("authToken", token);
  document.cookie = `authToken=${token}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
  document.cookie = `token=${token}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
}

export function removeAuthToken(): void {
  localStorage.removeItem("token");
  localStorage.removeItem("authToken");
  localStorage.removeItem("auth_token");
  document.cookie = "authToken=; path=/; max-age=0; SameSite=Lax";
  document.cookie = "token=; path=/; max-age=0; SameSite=Lax";
}

